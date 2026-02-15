#!/usr/bin/env python3
"""Test config + database connection."""
import os
from dotenv import load_dotenv

# Create app/__init__.py if missing
os.makedirs("app", exist_ok=True)
if not os.path.exists("app/__init__.py"):
    open("app/__init__.py", "w").close()

# Load .env
load_dotenv()

# Test imports
from app.core.config import settings
from app.core.database import Database

print("📋 Testing Configuration...")
print(f"✓ Supabase URL: {settings.supabase_url[:30]}...")
print(f"✓ Model path: {settings.model_path}")
print(f"✓ Debug mode: {settings.debug}")

print("\n🧪 Testing Database Connection...")
try:
    # Test visit_id=1 (from your sample data)
    features = Database.get_visit_features(1)
    print("✅ SUCCESS! Database connected.")
    print("Sample features:", {k: v for k, v in features.items() if k in ['age', 'heart_rate', 'chief_complaint']})
except Exception as e:
    print("❌ FAILED:", str(e))
    print("💡 Make sure you ran the SQL function in Supabase first!")
