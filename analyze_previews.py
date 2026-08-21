#!/usr/bin/env python3
"""Resolve exact Apple/iTunes previews and extract reproducible audio-signal evidence."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import subprocess
import time
import unicodedata
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


VERSION_WORDS = {
    "remix", "live", "karaoke", "instrumental", "sped", "slowed", "acoustic",
    "edit", "mix", "version", "remaster", "demo", "cover",
}

DEFAULT_COUNTRIES = ("US", "GB", "CA", "AU", "KR", "JP", "DE", "FR")


def urlopen_retry(request: urllib.request.Request, timeout: int, attempts: int = 3):
    """Open transient-prone preview endpoints with bounded exponential retry."""
    last_error = None
    for attempt in range(attempts):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.75 * (2 ** attempt))
    raise last_error


def norm(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode()
    value = re.sub(r"\b(feat|ft|featuring)\.?\b.*", "", value, flags=re.I)
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def version_penalty(requested: str, candidate: str) -> float:
    req = set(norm(requested).split()) & VERSION_WORDS
    got = set(norm(candidate).split()) & VERSION_WORDS
    return 0.0 if got <= req else min(0.35, 0.12 * len(got - req))


def search_track(artist: str, title: str, countries: tuple[str, ...]) -> dict:
    attempts = []
    best = None
    query_terms = (f"{artist} {title}", f"{title} {artist}")
    for country in countries:
        for term in query_terms:
            query = urllib.parse.urlencode({
                "term": term, "entity": "song", "limit": 25, "country": country,
            })
            req = urllib.request.Request(
                f"https://itunes.apple.com/search?{query}",
                headers={"User-Agent": "MusicReferenceAnalyzer/2.0"},
            )
            try:
                with urlopen_retry(req, timeout=25, attempts=3) as response:
                    results = json.load(response).get("results", [])
            except Exception as exc:
                attempts.append({"country": country, "term": term, "error": f"{type(exc).__name__}: {exc}"})
                continue
            ranked = []
            for item in results:
                if not item.get("previewUrl"):
                    continue
                title_sim = similarity(title, item.get("trackName", ""))
                artist_sim = similarity(artist, item.get("artistName", ""))
                score = 0.62 * title_sim + 0.38 * artist_sim - version_penalty(title, item.get("trackName", ""))
                ranked.append((score, title_sim, artist_sim, item))
            attempts.append({"country": country, "term": term, "candidates": len(ranked)})
            if not ranked:
                continue
            candidate = max(ranked, key=lambda row: row[0])
            if best is None or candidate[0] > best[0]:
                best = (*candidate, country)
            score, title_sim, artist_sim, item = candidate
            if score >= 0.82 and title_sim >= 0.84 and artist_sim >= 0.65:
                best = (*candidate, country)
                break
        if best and best[0] >= 0.82 and best[1] >= 0.84 and best[2] >= 0.65:
            break
    if best is None:
        return {"status": "AUDIO_UNAVAILABLE", "reason": "no preview candidate", "attempts": attempts}
    score, title_sim, artist_sim, item, country = best
    status = "MATCHED" if score >= 0.82 and title_sim >= 0.84 and artist_sim >= 0.65 else "REVIEW_MATCH"
    return {
        "status": status,
        "match_score": round(score, 4),
        "title_similarity": round(title_sim, 4),
        "artist_similarity": round(artist_sim, 4),
        "matched_artist": item.get("artistName"),
        "matched_title": item.get("trackName"),
        "collection": item.get("collectionName"),
        "release_date": item.get("releaseDate"),
        "track_view_url": item.get("trackViewUrl"),
        "preview_url": item.get("previewUrl"),
        "country": country,
        "attempts": attempts,
        "source": "Apple/iTunes official preview API",
    }


def download_preview(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "MusicReferenceAnalyzer/2.0"})
    with urlopen_retry(req, timeout=45, attempts=3) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 256):
            output.write(chunk)
    if destination.stat().st_size < 10_000:
        destination.unlink(missing_ok=True)
        raise ValueError("downloaded preview is unexpectedly small")


def convert_to_wav(source: Path, destination: Path) -> None:
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-ar", "44100", "-ac", "2", str(destination),
    ], check=True)


def scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def band_ratio(power: np.ndarray, freqs: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    total = float(power.sum()) + 1e-12
    return float(power[mask].sum() / total)


def key_estimate(chroma: np.ndarray) -> dict:
    # Krumhansl-Schmuckler key profiles; output is an estimate, never a verified score fact.
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    pitch = np.mean(chroma, axis=1)
    pitch = (pitch - pitch.mean()) / (pitch.std() + 1e-9)
    labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    candidates = []
    for root in range(12):
        for mode, template in (("major", major), ("minor", minor)):
            rolled = np.roll(template, root)
            corr = float(np.corrcoef(pitch, rolled)[0, 1])
            candidates.append((corr, labels[root], mode))
    candidates.sort(reverse=True)
    best, second = candidates[0], candidates[1]
    return {"key": best[1], "scale": best[2], "strength": round(best[0], 4), "margin": round(best[0] - second[0], 4)}


def segment_metrics(y: np.ndarray, sr: int, onset_env: np.ndarray, hop: int, start: float, end: float) -> dict:
    a, b = int(start * sr), min(len(y), int(end * sr))
    clip = y[a:b]
    if len(clip) < 32:
        return {"start": start, "end": end, "available": False}
    rms = librosa.feature.rms(y=clip, frame_length=2048, hop_length=hop)[0]
    c = librosa.feature.spectral_centroid(y=clip, sr=sr, hop_length=hop)[0]
    onset_a, onset_b = int(start * sr / hop), int(end * sr / hop)
    oe = onset_env[onset_a:onset_b]
    return {
        "start": start, "end": round(min(end, len(y) / sr), 3), "available": True,
        "rms_db_mean": round(float(librosa.amplitude_to_db(np.maximum(rms, 1e-9)).mean()), 3),
        "spectral_centroid_hz": round(float(c.mean()), 2),
        "onset_strength_mean": round(float(oe.mean()) if len(oe) else 0.0, 4),
        "onset_strength_cv": round(float(oe.std() / (oe.mean() + 1e-9)) if len(oe) else 0.0, 4),
    }


def analyze_wav(path: Path) -> dict:
    stereo, sr = sf.read(path, always_2d=True)
    y = stereo.mean(axis=1).astype(np.float32)
    duration = len(y) / sr
    hop = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop)
    tempo = scalar(tempo)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop, backtrack=False)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-9), ref=1.0)
    peak = float(np.max(np.abs(stereo)))
    crest = 20 * math.log10((peak + 1e-9) / (float(np.sqrt(np.mean(stereo ** 2))) + 1e-9))
    stft = librosa.stft(y, n_fft=4096, hop_length=hop)
    power = np.abs(stft) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    harmonic, percussive = librosa.effects.hpss(y)
    h_rms = float(np.sqrt(np.mean(harmonic ** 2)))
    p_rms = float(np.sqrt(np.mean(percussive ** 2)))
    centroid = librosa.feature.spectral_centroid(S=np.abs(stft), sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=np.abs(stft), sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(S=np.abs(stft))[0]

    pulse_clarity = 0.0
    if len(onset_env) > 8 and tempo > 0:
        ac = librosa.autocorrelate(onset_env, max_size=len(onset_env) // 2)
        lag = int(round((60.0 / tempo) * sr / hop))
        if 0 < lag < len(ac):
            pulse_clarity = float(ac[lag] / (ac[0] + 1e-9))

    offbeat_ratio = None
    if len(beats) >= 3 and len(onset_frames):
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop)
        interval = float(np.median(np.diff(beat_times)))
        distances = []
        for t in onset_times:
            phase = ((t - beat_times[0]) / max(interval, 1e-6)) % 1.0
            distances.append(min(abs(phase), abs(phase - 0.5), abs(phase - 1.0)))
        # Events nearer the half-beat grid than the beat grid indicate subdivision/offbeat activity.
        beat_dist = [min(((t - beat_times[0]) / max(interval, 1e-6)) % 1.0, 1 - (((t - beat_times[0]) / max(interval, 1e-6)) % 1.0)) for t in onset_times]
        half_dist = [abs((((t - beat_times[0]) / max(interval, 1e-6)) % 1.0) - 0.5) for t in onset_times]
        offbeat_ratio = float(np.mean(np.array(half_dist) < np.array(beat_dist)))

    stereo_corr = None
    side_mid = None
    if stereo.shape[1] >= 2:
        left, right = stereo[:, 0], stereo[:, 1]
        stereo_corr = float(np.corrcoef(left, right)[0, 1]) if left.std() and right.std() else 1.0
        mid = (left + right) / 2
        side = (left - right) / 2
        side_mid = float(np.sqrt(np.mean(side ** 2)) / (np.sqrt(np.mean(mid ** 2)) + 1e-9))

    segments = [
        segment_metrics(y, sr, onset_env, hop, 0, 5),
        segment_metrics(y, sr, onset_env, hop, 5, 10),
        segment_metrics(y, sr, onset_env, hop, 10, min(30, duration)),
    ]
    carry_delta = None
    if segments[1].get("available") and segments[2].get("available"):
        carry_delta = round(segments[2]["rms_db_mean"] - segments[1]["rms_db_mean"], 3)

    return {
        "scope": "PREVIEW_SEGMENT_NOT_SONG_INTRO",
        "duration_seconds": round(duration, 3),
        "sample_rate": sr,
        "tempo_bpm_estimate": round(tempo, 3),
        "beat_count": int(len(beats)),
        "pulse_clarity": round(pulse_clarity, 4),
        "onsets_per_second": round(len(onset_frames) / duration, 4),
        "offbeat_subdivision_ratio_estimate": None if offbeat_ratio is None else round(offbeat_ratio, 4),
        "key_estimate": key_estimate(chroma),
        "loudness_proxy": {
            "rms_db_mean": round(float(rms_db.mean()), 3),
            "rms_db_p10": round(float(np.percentile(rms_db, 10)), 3),
            "rms_db_p90": round(float(np.percentile(rms_db, 90)), 3),
            "rms_dynamic_range_db": round(float(np.percentile(rms_db, 90) - np.percentile(rms_db, 10)), 3),
            "sample_peak_dbfs": round(20 * math.log10(peak + 1e-12), 3),
            "crest_factor_db": round(crest, 3),
        },
        "spectrum": {
            "sub_20_80_ratio": round(band_ratio(power, freqs, 20, 80), 5),
            "bass_80_250_ratio": round(band_ratio(power, freqs, 80, 250), 5),
            "low_mid_250_500_ratio": round(band_ratio(power, freqs, 250, 500), 5),
            "mid_500_2000_ratio": round(band_ratio(power, freqs, 500, 2000), 5),
            "presence_2000_6000_ratio": round(band_ratio(power, freqs, 2000, 6000), 5),
            "air_6000_20000_ratio": round(band_ratio(power, freqs, 6000, 20000), 5),
            "centroid_hz_mean": round(float(centroid.mean()), 2),
            "bandwidth_hz_mean": round(float(bandwidth.mean()), 2),
            "flatness_mean": round(float(flatness.mean()), 5),
        },
        "texture": {
            "harmonic_to_percussive_rms_ratio": round(h_rms / (p_rms + 1e-9), 4),
            "zero_crossing_rate": round(float(librosa.feature.zero_crossing_rate(y)[0].mean()), 5),
        },
        "stereo": {
            "channel_correlation": None if stereo_corr is None else round(stereo_corr, 4),
            "side_to_mid_rms_ratio": None if side_mid is None else round(side_mid, 4),
        },
        "preview_windows": segments,
        "carry_energy_delta_db_5_10_to_10_30": carry_delta,
        "limits": [
            "Preview offset may not equal the song's opening 0-30 seconds.",
            "Full-song form, payoff, fatigue and replay cannot be confirmed from this preview alone.",
            "Tempo and key are signal estimates, not label-supplied facts.",
        ],
    }


def process(track: dict, cache: Path, countries: tuple[str, ...]) -> dict:
    artist, title = track["artist"], track["title"]
    result = {"artist": artist, "title": title, "pool": track.get("pool")}
    try:
        match = search_track(artist, title, countries)
        result["route"] = match
        if match["status"] != "MATCHED":
            result["audio_status"] = match["status"]
            return result
        digest = hashlib.sha256(f"{artist}|{title}|{match['preview_url']}".encode()).hexdigest()[:16]
        source = cache / f"{digest}.m4a"
        wav = cache / f"{digest}.wav"
        analysis_cache = cache / f"{digest}.analysis.json"
        if not source.exists():
            download_preview(match["preview_url"], source)
        if not wav.exists():
            convert_to_wav(source, wav)
        result["audio_status"] = "ANALYZED_PREVIEW"
        if analysis_cache.exists():
            result["audio"] = json.loads(analysis_cache.read_text(encoding="utf-8"))
        else:
            result["audio"] = analyze_wav(wav)
            analysis_cache.write_text(json.dumps(result["audio"], ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        result["audio_status"] = "AUDIO_UNAVAILABLE"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def metric_vector(row: dict) -> dict:
    audio = row.get("audio") or {}
    loud = audio.get("loudness_proxy") or {}
    spectrum = audio.get("spectrum") or {}
    texture = audio.get("texture") or {}
    stereo = audio.get("stereo") or {}
    return {
        "tempo_bpm": audio.get("tempo_bpm_estimate"),
        "pulse_clarity": audio.get("pulse_clarity"),
        "onsets_per_second": audio.get("onsets_per_second"),
        "offbeat_ratio": audio.get("offbeat_subdivision_ratio_estimate"),
        "dynamic_range_db": loud.get("rms_dynamic_range_db"),
        "crest_factor_db": loud.get("crest_factor_db"),
        "sub_ratio": spectrum.get("sub_20_80_ratio"),
        "bass_ratio": spectrum.get("bass_80_250_ratio"),
        "presence_ratio": spectrum.get("presence_2000_6000_ratio"),
        "air_ratio": spectrum.get("air_6000_20000_ratio"),
        "spectral_centroid_hz": spectrum.get("centroid_hz_mean"),
        "spectral_flatness": spectrum.get("flatness_mean"),
        "harmonic_percussive_ratio": texture.get("harmonic_to_percussive_rms_ratio"),
        "stereo_side_mid_ratio": stereo.get("side_to_mid_rms_ratio"),
        "carry_energy_delta_db": audio.get("carry_energy_delta_db_5_10_to_10_30"),
    }


def summarize_metrics(rows: list[dict]) -> dict:
    vectors = [metric_vector(row) for row in rows if row.get("audio_status") == "ANALYZED_PREVIEW"]
    if not vectors:
        return {}
    output = {}
    for name in vectors[0]:
        values = [v[name] for v in vectors if v.get(name) is not None and math.isfinite(v[name])]
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        output[name] = {
            "n": len(values), "median": round(float(np.median(arr)), 4),
            "q1": round(float(np.percentile(arr, 25)), 4),
            "q3": round(float(np.percentile(arr, 75)), 4),
            "min": round(float(arr.min()), 4), "max": round(float(arr.max()), 4),
        }
    return output


def validate_pool_design(tracks: list[dict]) -> dict:
    aliases = {
        "COMMERCIAL": "COMMERCIAL",
        "PLAYLIST": "PLAYLIST_LONGEVITY",
        "PLAYLIST_LONGEVITY": "PLAYLIST_LONGEVITY",
        "TREND": "CURRENT_TREND",
        "CURRENT_TREND": "CURRENT_TREND",
    }
    counts = {name: 0 for name in ("COMMERCIAL", "PLAYLIST_LONGEVITY", "CURRENT_TREND")}
    artist_counts = {name: {} for name in counts}
    unique = set()
    for track in tracks:
        pool = aliases.get(str(track.get("pool", "")).upper())
        if not pool:
            continue
        counts[pool] += 1
        artist = norm(track.get("artist", ""))
        artist_counts[pool][artist] = artist_counts[pool].get(artist, 0) + 1
        unique.add((artist, norm(track.get("title", ""))))
    violations = []
    for pool, count in counts.items():
        if count < 10:
            violations.append(f"{pool} has {count}; requires at least 10")
        over = {artist: n for artist, n in artist_counts[pool].items() if n > 2}
        if over:
            violations.append(f"{pool} artist limit exceeded: {over}")
        if len(artist_counts[pool]) < 5:
            violations.append(f"{pool} has {len(artist_counts[pool])} artists; requires at least 5")
    if len(unique) < 30:
        violations.append(f"unique tracks {len(unique)}; target at least 30")
    return {"status": "PASS" if not violations else "HOLD", "pool_counts": counts, "unique_tracks": len(unique), "violations": violations}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, help="JSON file: {tracks:[{artist,title,pool}]} or a raw list")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/music-reference-audio"))
    parser.add_argument("--countries", default=",".join(DEFAULT_COUNTRIES),
                        help="Comma-separated Apple storefront retry order")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    tracks = data.get("tracks", data) if isinstance(data, dict) else data
    countries = tuple(x.strip().upper() for x in args.countries.split(",") if x.strip())
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(args.workers, 6))) as executor:
        futures = [executor.submit(process, track, args.cache, countries) for track in tracks]
        results = [future.result() for future in futures]
    counts = {}
    for row in results:
        counts[row["audio_status"]] = counts.get(row["audio_status"], 0) + 1
    pool_rows = {}
    for row in results:
        pool_rows.setdefault(row.get("pool") or "UNSPECIFIED", []).append(row)
    pool_gate = validate_pool_design(tracks)
    audio_coverage = counts.get("ANALYZED_PREVIEW", 0) / max(len(results), 1)
    report = {
        "method": "Multi-store exact-match Apple/iTunes official preview routing plus direct DSP signal analysis",
        "countries_attempted": list(countries),
        "disclaimer": "This is automated analysis of actual preview audio, not human-like model listening and not full-song analysis.",
        "count": len(results), "status_counts": counts,
        "pool_gate": pool_gate,
        "audio_coverage": round(audio_coverage, 4),
        "reference_audio_gate": "PASS" if pool_gate["status"] == "PASS" and audio_coverage >= 0.8 else "HOLD",
        "statistics": {
            "all": summarize_metrics(results),
            "by_pool": {pool: summarize_metrics(rows) for pool, rows in pool_rows.items()},
        },
        "tracks": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "count": len(results), "status_counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
