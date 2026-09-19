# Data contract

Raw observations are intentionally not included in this repository.

Expected inputs are:

- HSR and TSR APE targets derived separately from 350-1050 nm spectra;
- ground-observed air mass (`AM`);
- MERRA-2 total precipitable water (`TQV`);
- MERRA-2 aerosol optical depth at 550 nm (`TOTEXTTAU` / `AOD550`);
- MERRA-2 Angstrom exponent (`TOTANGSTR`);
- MERRA-2 total-column ozone (`TO3`).

Times must be normalized to UTC before exact hourly-center alignment. HSR and TSR remain separate rows, and duplicate site-times must not be silently averaged. Input paths are supplied through command-line arguments; no personal absolute paths are stored in the repository.

Users are responsible for obtaining the source datasets and complying with their providers' licenses and institutional handling rules.
