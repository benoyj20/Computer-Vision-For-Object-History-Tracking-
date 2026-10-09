# Local data

Everything in this directory except this file is ignored by Git. Set `OBJHIST_DATA_DIR` in
`.env` to keep data elsewhere. Record sources, versions, and hashes in
[the data card](../docs/data-card.md).

Suggested layout:

```text
data/
├── raw/          # downloads and provided sequences exactly as received
│   ├── roboflow-kitchen-items/
│   └── sequences/<session-id>/
├── processed/    # YOLO-format datasets ready for training
│   └── kitchen/
│       ├── data.yaml
│       ├── images/{train,val,test}/
│       └── labels/{train,val,test}/
├── scenarios/    # scripted evaluation sequences and their ground truth
└── histories/    # generated object histories (SQLite)
```

## Provided snapshot sequences

The project does not capture images itself; Benoy provides them. Each session is one folder of
snapshots from a single fixed camera position, plus a manifest:

```text
data/raw/sequences/<session-id>/
├── manifest.csv
├── 000001.jpg
├── 000002.jpg
└── ...
```

`manifest.csv` has one row per image, in capture order:

```csv
filename,captured_at
000001.jpg,2026-10-12T14:05:00-04:00
000002.jpg,2026-10-12T14:06:00-04:00
```

- `captured_at` is ISO 8601 with a UTC offset. Placement intervals and "how long has it been
  out" answers come from these times, so they must be the real capture times, not file
  modification times.
- The camera does not move within a session. If it moves, start a new session.
- Images within a session have the same resolution.
- Use a short, descriptive session ID such as `2026-10-12-evening`. A session belongs to exactly
  one split (train, validation, or test).
- For scripted evaluation scenarios, also include the list of changes that were made (what,
  where, and when) so ground truth can be annotated; see
  [the evaluation protocol](../docs/evaluation-protocol.md).

If capture times are only in the images' EXIF data, deliver the images and the manifest can be
generated from EXIF, provided the camera clock and time zone were correct.

## Training

Train with the dataset's `data.yaml`, for example
`make train DATA=data/processed/kitchen/data.yaml`. Keep the class order in `data.yaml`
identical to the class list in the data card. Ultralytics writes training output to `runs/`,
which is also ignored.
