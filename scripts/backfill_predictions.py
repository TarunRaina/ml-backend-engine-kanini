"""
Backfill script to populate triage_predictions for existing patient visits.
This is a ONE-TIME operation to fill historical data.
"""
import sys
import os
from tqdm import tqdm
import time

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from app.core.database import Database
from app.models.ml_engine import MLEngine

def backfill_predictions():
    """Backfill triage predictions for all existing visits."""
    print("🚀 Starting backfill process...")
    
    # Initialize
    db = Database()
    client = db.get_client()
    ml_engine = MLEngine(model_dir='app/models')
    
    # 1. Get all visit IDs from patient_visits
    print("📊 Fetching all patient visits...")
    response = client.table('patient_visits').select('visit_id').execute()
    visits = response.data
    
    if not visits:
        print("❌ No visits found in database!")
        return
    
    print(f"✅ Found {len(visits)} visits to process")
    
    # 2. Check which visits already have predictions
    print("🔍 Checking existing predictions...")
    existing_response = client.table('triage_predictions').select('visit_id').execute()
    existing_visit_ids = {row['visit_id'] for row in existing_response.data}
    
    # Filter out visits that already have predictions
    visits_to_process = [v for v in visits if v['visit_id'] not in existing_visit_ids]
    
    print(f"📝 {len(existing_visit_ids)} visits already have predictions")
    print(f"🎯 {len(visits_to_process)} visits need predictions")
    
    if not visits_to_process:
        print("✅ All visits already have predictions! Nothing to do.")
        return
    
    # 3. Process each visit
    success_count = 0
    error_count = 0
    errors = []
    
    print("\n🔄 Processing visits...")
    for visit in tqdm(visits_to_process, desc="Backfilling"):
        visit_id = visit['visit_id']
        
        try:
            # Get visit features
            features = db.get_visit_features(visit_id)
            
            # Run ML prediction
            prediction = ml_engine.predict(features)
            
            # Save to database
            db.save_prediction(visit_id, prediction)
            
            success_count += 1
            
            # Small delay to avoid overwhelming the database
            time.sleep(0.1)
            
        except Exception as e:
            error_count += 1
            errors.append({'visit_id': visit_id, 'error': str(e)})
            print(f"\n⚠️ Error processing visit {visit_id}: {e}")
    
    # 4. Summary
    print("\n" + "="*60)
    print("📊 BACKFILL SUMMARY")
    print("="*60)
    print(f"✅ Successfully processed: {success_count}")
    print(f"❌ Errors: {error_count}")
    print(f"📈 Total visits in DB: {len(visits)}")
    print(f"📋 Predictions now in DB: {len(existing_visit_ids) + success_count}")
    
    if errors:
        print("\n⚠️ ERRORS:")
        for err in errors[:10]:  # Show first 10 errors
            print(f"  Visit {err['visit_id']}: {err['error']}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    print("\n✅ Backfill complete!")

if __name__ == "__main__":
    try:
        backfill_predictions()
    except KeyboardInterrupt:
        print("\n\n⚠️ Backfill interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
