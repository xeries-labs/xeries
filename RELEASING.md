# Releasing xeries

This guide describes how to release a new version of xeries to PyPI.

## Prerequisites

1. **PyPI Trusted Publishing configured**
   - Go to https://pypi.org/manage/project/xeries/settings/publishing/
   - Add a new publisher:
     - Owner: `xeries-labs`
     - Repository: `xeries`
     - Workflow name: `release.yml`
     - Environment name: `pypi`

2. **TestPyPI Trusted Publishing configured** (optional, for testing)
   - Go to https://test.pypi.org/manage/project/xeries/settings/publishing/
   - Same settings but environment name: `testpypi`

3. **GitHub Environments configured**
   - Create `pypi` environment in repo Settings > Environments
   - Create `testpypi` environment in repo Settings > Environments
   - Optionally add protection rules (require approval, etc.)

4. **Zenodo GitHub integration configured**
   - Connect repository in Zenodo: https://zenodo.org/account/settings/github/
   - Enable archiving for `xeries-labs/xeries`
   - Ensure `.zenodo.json` is present and up to date
   - Current DOI: https://doi.org/10.5281/zenodo.19482748

## Release Process

### 1. Update Version

Edit `src/xeries/_version.py` and update the version:

```python
__version__ = "0.2.0"  # New version
```

The version is read dynamically by `pyproject.toml` via hatchling.

Follow [Semantic Versioning](https://semver.org/):
- **MAJOR** (1.0.0): Breaking API changes
- **MINOR** (0.2.0): New features, backward compatible
- **PATCH** (0.1.1): Bug fixes, backward compatible

### 2. Update Changelog (if applicable)

Document changes in a CHANGELOG.md or release notes.

### 3. Commit and Push

```bash
git add src/xeries/_version.py
git commit -m "Bump version to 0.2.0"
git push origin main
```

### 4. Test Release (Optional)

Test the release on TestPyPI first:

1. Go to Actions > Release workflow
2. Click "Run workflow"
3. Select `testpypi` as target
4. Click "Run workflow"

Verify at https://test.pypi.org/project/xeries/

Test installation:
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/xeries
```

### 5. Create GitHub Release (Recommended)

Create the release in GitHub UI (this will create and push the `v0.2.0` tag automatically):

1. Go to GitHub > Releases > Draft a new release
2. Choose tag `v0.2.0` (or create a new tag with that name)
3. Set release title (for example, `xeries v0.2.0`)
4. Publish release

Publishing the release automatically triggers:
1. Package build
2. Version validation (tag must match pyproject.toml)
3. PyPI publication
4. Release artifact upload/update

Alternative (CLI tag flow):

```bash
git tag v0.2.0
git push origin v0.2.0
```

### 6. Verify Release

- Check https://pypi.org/project/xeries/
- Check GitHub Releases page
- Check Zenodo record/DOI (new version should be archived automatically)
- Test installation: `pip install xeries==0.2.0`

## Manual Release (Emergency)

If you need to publish without creating a tag:

1. Go to Actions > Release workflow
2. Click "Run workflow"
3. Select `pypi` as target
4. Click "Run workflow"

**Warning**: This bypasses version validation. Ensure pyproject.toml version is correct.

## Troubleshooting

### "Tag version does not match package version"

The git tag (e.g., `v0.2.0`) must match the version in `src/xeries/_version.py` (e.g., `0.2.0`).

Fix:
```bash
# Delete wrong tag
git tag -d v0.2.0
git push origin :refs/tags/v0.2.0

# Fix version in src/xeries/_version.py, commit, then re-tag
git tag v0.2.0
git push origin v0.2.0
```

### "Project not found" on PyPI

Trusted publishing requires the project to exist. For the first release:
1. Build locally: `uv build`
2. Upload manually: `uv publish` (requires API token)
3. Then configure trusted publishing

### Build Fails

Run locally to debug:
```bash
uv build
python -m zipfile -l dist/*.whl
```

Check that `src/xeries/` is included in the wheel.
