# Audio bridge verification

Date: 2026-08-21

- Direct engine, 3 tracks: 3/3 `ANALYZED_PREVIEW`
- GPT-Action-shaped create/poll/summary/tracks round trip: 3/3 `ANALYZED_PREVIEW`
- Direct engine, 30 tracks: 28/30 `ANALYZED_PREVIEW`, 2/30 `REVIEW_MATCH`
- Actual preview coverage: 93.3%
- Pool design: Commercial 10 / Playlist-Longevity 10 / Current-Trend 10, PASS
- Rejected cases: one wrong artist with the same title; one unwanted `Mixed` version
- Storefront retry order: US, GB, CA, AU, KR, JP, DE, FR

The test processed official Apple/iTunes preview bytes. It did not treat a streaming page or playback link as audio evidence. Preview offsets were not relabeled as true song openings.
