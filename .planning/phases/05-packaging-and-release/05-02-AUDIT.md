# GitHub Repository Configuration Audit

**Audited:** 2026-02-19T01:27:07Z
**Repository:** tlancaster6/AquaCore (https://github.com/tlancaster6/AquaCore)

## Secrets (`gh secret list`)

| Secret | Status |
|--------|--------|
| RELEASE_TOKEN | MISSING |
| CODECOV_TOKEN | MISSING (optional) |

*No secrets configured on this repository.*

## Environments (`gh api repos/tlancaster6/aquacore/environments`)

| Environment | Status |
|-------------|--------|
| testpypi | MISSING |
| pypi | MISSING |

*`total_count: 0` — no environments configured.*

## Branch Protection on `main`

| Rule | Status |
|------|--------|
| Branch protection exists | MISSING |
| Required status checks | MISSING |
| Require PR before merging | MISSING |

*API returned 404 "Branch not protected" — no rules configured.*

## Summary Checklist

- [ ] Create RELEASE_TOKEN PAT (repo scope) and add as repository secret
- [ ] Create CODECOV_TOKEN secret (optional)
- [ ] Create `testpypi` GitHub environment
- [ ] Create `pypi` GitHub environment (optionally add required reviewers)
- [ ] Register trusted publisher on PyPI for publish.yml + environment=pypi
- [ ] Register trusted publisher on TestPyPI for publish.yml + environment=testpypi
- [ ] Configure branch protection on `main` (after CI has run at least once on main)
