# Corpus manifest

`corpus_manifest.csv` maps the local PDF inventory to the Dubai Legislation Portal URL pattern. It currently contains all 45 PDFs in `data/raw/pdf/` as candidate records.

## Portal source and category

- Official search page: https://dlp.dubai.gov.ae/en/Pages/LegislationSearch.aspx
- Exact primary-category filter to use: `Roads and Transport`.
- Detail and content URLs in the CSV use the portal's `Legislation Reference/{year}/{title}.html|pdf` URL pattern. Several were opened successfully; abbreviated local filenames can differ from the official displayed title. Check each link and correct any title/path mismatches before treating the manifest as final.

## Fields needing review

- `source_detail_url` and `content_url`: generated using each local filename; verify they open the matching legislation record.
- `title`: currently the local PDF filename stem; replace with the portal's full official title after matching.
- `primary_category`: populated with the requested category as the intended filter, not as a claim that every local file has been individually verified in the portal results.
- `language`: expected English corpus; confirm each PDF is the English version.
- `retrieved_at`: date this manifest's portal source links were checked, not necessarily the date the local PDFs were downloaded.
- `issuing_authority` and `status`: blank unless confirmed from the portal.
- `category_verification`: record-level category confirmation is still required.

Before submission, save a screenshot or export of the portal results with the exact `Roads and Transport` primary filter applied, then keep it under `data/evidence/`. No such screenshot/export was present in the workspace when this manifest was created.

