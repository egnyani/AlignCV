"""
Simulate ATS scores for proposed bullet texts against all JD requirements.
Usage: python3 test_bullet.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pipeline.embedder import embed_texts
from pipeline.scorer import score_resume_against_jd, _keyword_score, _extract_keywords

REQUIREMENTS = [
    "5+ years experience building data pipelines and warehouse models",
    "Strong proficiency in Python and SQL for data processing",
    "Hands-on experience with Airflow or similar orchestration tools",
    "Experience with cloud data platforms like AWS or Snowflake",
    "Familiarity with distributed processing technologies such as Spark",
    "Ability to improve data reliability and pipeline performance",
    "Communicate clearly with technical and non-technical stakeholders",
    "Experience supporting product analytics or SaaS business intelligence",
    "Familiarity with dbt, Terraform, or infrastructure-as-code workflows",
    "Experience designing data quality frameworks and validation checks",
    "Comfortable mentoring junior engineers and driving best practices",
]

# Proposed bullets to test (edit these to experiment)
TEST_BULLETS = {
    "Airflow bullet":
        "Designed and maintained production-grade Airflow DAGs and orchestration workflows to schedule batch ELT pipelines, processing millions of records daily with automated alerting and retry logic.",
    "AWS/Snowflake bullet":
        "Contributed to cloud data platform architecture leveraging AWS S3 data lake storage, Snowflake warehousing, and CI/CD pipeline automation to power product analytics and executive dashboards.",
    "Spark bullet":
        "Built distributed data processing pipelines using Apache Spark and AWS EKS to process multi-terabyte workloads, enabling horizontally scalable data reliability for production analytics.",
    "Communicate bullet":
        "Partnered with technical and non-technical stakeholders—product, finance, and executive teams—to translate ambiguous data requirements into trusted datasets and self-service analytics products.",
    "IaC bullet":
        "Implemented infrastructure-as-code workflows using Terraform and CloudFormation, enabling reproducible data platform deployments and CI/CD pipeline automation with dbt model versioning.",
    "Data quality bullet":
        "Designed data quality frameworks and automated validation checks using dbt tests and Great Expectations, enabling root-cause analysis and observability through alerting and documentation.",
    "Mentoring bullet":
        "Mentored junior engineers in data engineering best practices—pipeline design, SQL optimization, and testing—while driving team-wide adoption of data quality and observability standards.",
}

def test_bullet(bullet_text: str, req_embeddings: list, requirements: list[str]) -> None:
    bullet_embs = embed_texts([bullet_text])
    bullet_emb = bullet_embs[0]
    import numpy as np
    scores = []
    for i, (req, req_emb) in enumerate(zip(requirements, req_embeddings)):
        # Cosine similarity
        a = np.array(bullet_emb)
        b = np.array(req_emb)
        sem_score = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        kw_score = _keyword_score(bullet_text, req)
        combined = 0.6 * kw_score + 0.4 * sem_score
        scores.append((combined, kw_score, sem_score, req))
    scores.sort(reverse=True)
    return scores

print("Loading requirement embeddings...")
req_embeddings = embed_texts(REQUIREMENTS)
print(f"Loaded {len(req_embeddings)} requirement embeddings.\n")

for label, bullet in TEST_BULLETS.items():
    print(f"{'='*70}")
    print(f"BULLET: [{label}]")
    print(f"  {bullet[:100]}...")
    scores = test_bullet(bullet, req_embeddings, REQUIREMENTS)
    print(f"  Best match: combined={scores[0][0]:.3f} kw={scores[0][1]:.2f} sem={scores[0][2]:.3f}")
    print(f"    → {scores[0][3][:70]}")
    if len(scores) > 1:
        print(f"  2nd best:   combined={scores[1][0]:.3f} kw={scores[1][1]:.2f} sem={scores[1][2]:.3f}")
        print(f"    → {scores[1][3][:70]}")

print("\n" + "="*70)
print("SCORE MATRIX (bullet vs requirement)")
print("="*70)
print(f"{'Requirement':<55} ", end="")
for label in TEST_BULLETS:
    short = label[:8]
    print(f"{short:>9}", end="")
print()
for i, req in enumerate(REQUIREMENTS):
    print(f"{req[:55]:<55} ", end="")
    for label, bullet in TEST_BULLETS.items():
        bullet_emb = embed_texts([bullet])[0]
        import numpy as np
        a = np.array(bullet_emb)
        b = np.array(req_embeddings[i])
        sem = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        kw = _keyword_score(bullet, req)
        combined = 0.6 * kw + 0.4 * sem
        marker = "✓" if combined >= 0.70 else " "
        print(f" {combined:.2f}{marker}", end="")
    print()
