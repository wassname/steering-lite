# Corrected stage label

The 429 reconciliation artifact named the affected stage `random` incorrectly.

The cached `random` condition had already completed: its `final-judgments` cache record `108bb8206e09` exists and the 9-second random-only rerun made no provider call. Reconstructing the failed payload from each cached 84-item final plan finds payload `334a6e70…` only in `mean_diff`, at request key `b91f8b…`.

The reservation `48d338…` and the conservative upper estimate remain correct. The only affected stage to retry is `mean_diff` final judgment work; all later methods remain stopped.

-- PI[gpt-5.6-terra]
