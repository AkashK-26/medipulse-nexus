# Step 5: ETL Pipeline and Setup Script
# setup.py

import os
import subprocess
import sys

def setup_hospital_analytics():
    """Complete setup script for Hospital Analytics Dashboard"""
    
    print("🏥 Hospital Operations Analytics Dashboard - Setup")
    print("=" * 60)
    
    # 1. Install dependencies
    print("\n📦 Installing dependencies...")
    dependencies = [
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "sqlalchemy==2.0.23",
        "psycopg2-binary==2.9.9",
        "pandas==2.1.3",
        "numpy==1.26.2",
        "prophet==1.1.5",
        "python-multipart==0.0.6",
        "jinja2==3.1.2",
        "pdfkit==1.0.0"
    ]
    
    for dep in dependencies:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", dep])
    
    # 2. Database setup instructions
    print("\n🗄️  Database Setup Instructions:")
    print("-" * 60)
    print("1. Install PostgreSQL:")
    print("   sudo apt-get install postgresql postgresql-contrib")
    print("\n2. Create database:")
    print("   sudo -u postgres psql")
    print("   CREATE DATABASE hospital_analytics;")
    print("   CREATE USER hospital_user WITH PASSWORD 'your_password';")
    print("   GRANT ALL PRIVILEGES ON DATABASE hospital_analytics TO hospital_user;")
    print("   \\q")
    print("\n3. Update DATABASE_URL in main.py with your credentials")
    
    # 3. Create directories
    print("\n📁 Creating directory structure...")
    os.makedirs("static/reports", exist_ok=True)
    os.makedirs("static/css", exist_ok=True)
    os.makedirs("static/js", exist_ok=True)
    
    # 4. Create .env template
    env_content = """# Hospital Analytics Configuration
DATABASE_URL=postgresql://hospital_user:password@localhost/hospital_analytics
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("\n✅ Setup complete!")
    print("\n🚀 To start the application:")
    print("   1. Update DATABASE_URL in main.py")
    print("   2. Run: python generate_sample_data.py")
    print("   3. Run: python main.py")
    print("   4. Open: http://localhost:8000")
    
    print("\n📊 Default dashboard credentials:")
    print("   URL: http://localhost:8000")
    print("   No login required for demo mode")

if __name__ == "__main__":
    setup_hospital_analytics()
