from __future__ import annotations

import concurrent.futures
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from analyze_previews import (
    DEFAULT_COUNTRIES,
    metric_vector,
    process,
    summarize_metrics,
    validate_pool_design,
)


class Track(BaseModel):
    pool: Literal["COMMERCIAL", "PLAYLIST_LONGEVITY", "CURRENT_TREND"]
    artist: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=300)


class JobRequest(BaseModel):
    tracks: list[Track] = Field(min_length=1, max_length=40)
    countries: list[str] = Field(default_factory=lambda: list(DEFAULT_COUNTRIES), max_length=12)


class BatchRequest(BaseModel):
    """Stateless action request sized to complete inside one GPT Action call."""

    tracks: list[Track] = Field(min_length=1, max_length=5)
    countries: list[str] = Field(default_factory=lambda: list(DEFAULT_COUNTRIES), max_length=12)


app = FastAPI(
    title="Official Music Preview Analyzer",
    version="1.0.0",
    description="Resolves lawful official Apple/iTunes previews and returns direct DSP evidence for music-reference analysis.",
    servers=[{"url": "https://music-reference-analyzer.onrender.com"}],
)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
expected_key = os.getenv("ANALYZER_API_KEY", "").strip()
cache_root = Path(os.getenv("AUDIO_CACHE_DIR", "/data/audio-cache"))
job_root = Path(os.getenv("JOB_DIR", "/data/jobs"))
cache_root.mkdir(parents=True, exist_ok=True)
job_root.mkdir(parents=True, exist_ok=True)
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
job_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(os.getenv("MAX_JOBS", "1"))))


def authorize(key: str | None = Depends(api_key_header)) -> None:
    if expected_key and key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


def persist(job_id: str) -> None:
    with jobs_lock:
        payload = jobs[job_id]
        (job_root / f"{job_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def status_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        name = row.get("audio_status", "UNKNOWN")
        counts[name] = counts.get(name, 0) + 1
    return counts


def analyze_batch_rows(request: BatchRequest) -> dict:
    """Analyze up to five previews synchronously; no job id or persistent disk required."""

    tracks = [track.model_dump() for track in request.tracks]
    countries = tuple(country.upper() for country in request.countries if country.strip())
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, len(tracks))) as pool:
        rows = list(pool.map(lambda track: process(track, cache_root, countries), tracks))
    counts = status_counts(rows)
    return {
        "method": "Official Apple/iTunes preview bytes, exact-version validation, multi-store retry, and direct DSP",
        "scope": "PREVIEW_SEGMENTS_NOT_CONFIRMED_SONG_INTROS",
        "count": len(rows),
        "status_counts": counts,
        "audio_coverage": round(counts.get("ANALYZED_PREVIEW", 0) / max(len(rows), 1), 4),
        "countries_attempted": list(countries),
        "statistics": summarize_metrics(rows),
        "tracks": rows,
        "limits": [
            "Preview offset may differ from the true song opening.",
            "Full-song form, payoff, long fatigue and replay remain unconfirmed.",
            "DSP measures audio behavior; it does not independently prove aesthetic quality.",
        ],
    }


def run_job(job_id: str, request: JobRequest) -> None:
    tracks = [track.model_dump() for track in request.tracks]
    countries = tuple(country.upper() for country in request.countries if country.strip())
    with jobs_lock:
        jobs[job_id]["status"] = "RUNNING"
    persist(job_id)
    rows: list[dict | None] = [None] * len(tracks)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            pending = {
                pool.submit(process, track, cache_root, countries): index
                for index, track in enumerate(tracks)
            }
            for future in concurrent.futures.as_completed(pending):
                index = pending[future]
                try:
                    rows[index] = future.result()
                except Exception as exc:
                    track = tracks[index]
                    rows[index] = {
                        "artist": track["artist"], "title": track["title"],
                        "pool": track["pool"], "audio_status": "AUDIO_UNAVAILABLE",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                with jobs_lock:
                    jobs[job_id]["completed"] += 1
                    jobs[job_id]["status_counts"] = status_counts([x for x in rows if x])
                persist(job_id)
        final_rows = [row for row in rows if row]
        counts = status_counts(final_rows)
        pool_gate = validate_pool_design(tracks)
        analyzed = counts.get("ANALYZED_PREVIEW", 0)
        coverage = analyzed / max(len(final_rows), 1)
        by_pool: dict[str, list[dict]] = {}
        for row in final_rows:
            by_pool.setdefault(row.get("pool") or "UNSPECIFIED", []).append(row)
        report = {
            "method": "Official Apple/iTunes preview bytes, exact-version validation, multi-store retry, and direct DSP",
            "scope": "PREVIEW_SEGMENTS_NOT_CONFIRMED_SONG_INTROS",
            "count": len(final_rows),
            "status_counts": counts,
            "audio_coverage": round(coverage, 4),
            "countries_attempted": list(countries),
            "pool_gate": pool_gate,
            "statistics": {
                "all": summarize_metrics(final_rows),
                "by_pool": {name: summarize_metrics(items) for name, items in by_pool.items()},
            },
            "tracks": final_rows,
            "limits": [
                "Preview offset may differ from the true song opening.",
                "Full-song form, payoff, long fatigue and replay remain unconfirmed.",
                "DSP measures audio behavior; it does not independently prove aesthetic quality.",
            ],
        }
        with jobs_lock:
            jobs[job_id].update({"status": "COMPLETE", "report": report, "status_counts": counts})
        persist(job_id)
    except Exception as exc:
        with jobs_lock:
            jobs[job_id].update({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
        persist(job_id)


def get_job_or_404(job_id: str) -> dict:
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        path = job_root / f"{job_id}.json"
        if path.exists():
            job = json.loads(path.read_text(encoding="utf-8"))
            with jobs_lock:
                jobs[job_id] = job
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/health", operation_id="healthCheck")
def health() -> dict:
    return {"status": "ok", "service": "official-music-preview-analyzer"}


@app.get("/privacy", response_class=PlainTextResponse, operation_id="privacyPolicy")
def privacy() -> str:
    return (
        "This private analysis service receives artist names and track titles, "
        "downloads lawful official preview files for transient analysis, and stores job results temporarily. "
        "It does not request streaming credentials or extract protected subscription streams."
    )


@app.post(
    "/v1/analyze-batch",
    dependencies=[Depends(authorize)],
    operation_id="analyzeMusicPreviewBatch",
)
def analyze_batch(request: BatchRequest) -> dict:
    return analyze_batch_rows(request)


@app.post("/v1/jobs", dependencies=[Depends(authorize)], operation_id="createMusicAnalysisJob")
def create_job(request: JobRequest) -> dict:
    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id, "status": "QUEUED", "total": len(request.tracks),
            "completed": 0, "status_counts": {},
        }
    persist(job_id)
    job_executor.submit(run_job, job_id, request)
    return {"job_id": job_id, "status": "QUEUED", "total": len(request.tracks)}


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(authorize)], operation_id="getMusicAnalysisJobStatus")
def get_job_status(job_id: str) -> dict:
    job = get_job_or_404(job_id)
    return {key: job.get(key) for key in ("job_id", "status", "total", "completed", "status_counts", "error") if job.get(key) is not None}


@app.get("/v1/jobs/{job_id}/summary", dependencies=[Depends(authorize)], operation_id="getMusicAnalysisSummary")
def get_job_summary(job_id: str) -> dict:
    job = get_job_or_404(job_id)
    if job.get("status") != "COMPLETE":
        raise HTTPException(status_code=409, detail=f"Job status is {job.get('status')}")
    report = job["report"]
    return {
        "method": report["method"], "scope": report["scope"], "count": report["count"],
        "status_counts": report["status_counts"], "audio_coverage": report["audio_coverage"],
        "countries_attempted": report["countries_attempted"], "pool_gate": report["pool_gate"],
        "statistics": report["statistics"], "limits": report["limits"],
        "track_index": [
            {"index": index, "pool": row.get("pool"), "artist": row["artist"], "title": row["title"],
             "audio_status": row.get("audio_status"), "metrics": metric_vector(row)}
            for index, row in enumerate(report["tracks"])
        ],
    }


@app.get("/v1/jobs/{job_id}/tracks", dependencies=[Depends(authorize)], operation_id="getMusicAnalysisTracks")
def get_job_tracks(
    job_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(5, ge=1, le=10),
) -> dict:
    job = get_job_or_404(job_id)
    if job.get("status") != "COMPLETE":
        raise HTTPException(status_code=409, detail=f"Job status is {job.get('status')}")
    rows = job["report"]["tracks"]
    return {"offset": offset, "limit": limit, "total": len(rows), "tracks": rows[offset:offset + limit]}
