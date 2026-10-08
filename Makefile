PYTHON ?= python

.PHONY: export validate

export:
	$(PYTHON) scripts/export_web_data.py

# Structure, recomputation from results/, two-build determinism and hygiene.
validate:
	$(PYTHON) scripts/validate_web_export.py
