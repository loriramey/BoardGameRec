# Changelog
All notable changes to **Play Next** are documented here.  
This project follows [Semantic Versioning](https://semver.org/).

---
## [1.5.0] - 2025-Sept  WIP
### Added
- New directories such as pipelines/ and legacy/ created to organize files for future work

### Changed
- Files relevant for future data transformation have been moved to a pipelines/utils folder
- Files related to the capstone elements of the app have been moved to a /legacy folder 

### Deprecated
- Files related to preparing data visualizations or alternate cosine similarty and data processing recipes

### Removed
- Artifacts of data visualization and some .py files that were used only to export visuals required by my capstone paper

---
## [1.4.0] - 2025-Sept
### Changed
- Freeze main project files as they existed for 1.0 release. This step will lay groundwork for a v1.5 to prepare for shifting to API structure

## [1.0.0] - 2025-Apr
### Added
- Initial base app: cosine-similarity recommendations, fuzzy matching, filters, and data visualizations.
- Two versions exist of this app: 
  - This version is the "main" one with deeper files and more under-the-hood info about the underlying ML calculations

---
for future

## [2.0.0] - 2025-Sept
IN PROGRESS

### Added
- Explore page: search by game title and designer, with filters (players, weight, playtime, year, rating).
- Affiliate links (Amazon/eBay) on results for modern titles for easy transfer to online store pages

### Changed
- Fuzzy search overhaul: exact-match priority, multi-edition display `(Year)`, startswith boosting.

### Fixed
- Classic and popular titles (e.g., *Clue*, *Risk*, *Root*) now reliably appear in search.
- Rating display defaults to raw user average to match BGG expectations.

### Deprecated
- TBD

### Removed
- TBD

### Security
- TBD