"""Quick script to verify backfill results."""
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from app.core.database import Database

db = Database()
client = db.get_client()

# Count total visits
visits_response = client.table('patient_visits').select('visit_id', count='exact').execute()
total_visits = visits_response.count

# Count predictions
predictions_response = client.table('triage_predictions').select('prediction_id', count='exact').execute()
total_predictions = predictions_response.count

print("="*60)
print("BACKFILL VERIFICATION")
print("="*60)
print(f"Total patient visits: {total_visits}")
print(f"Total predictions: {total_predictions}")
print(f"Coverage: {(total_predictions/total_visits*100):.1f}%")
print("="*60)

# Show sample predictions
print("\nSample predictions:")
sample = client.table('triage_predictions').select('*').limit(5).execute()
for pred in sample.data:
    print(f"\nVisit {pred['visit_id']}:")
    print(f"  Risk: {pred['risk_level']} ({pred['risk_score']:.2f})")
    print(f"  Department: {pred['recommended_department']}")
