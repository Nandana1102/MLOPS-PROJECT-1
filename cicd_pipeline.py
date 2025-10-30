import os

print("🚀 Starting CI/CD Pipeline...")

# Step 1: Data pipeline
print("📊 Running data preprocessing...")
os.system("python src/data_pipeline.py")

# Step 2: Train model
print("🤖 Training model...")
os.system("python src/train_model.py")

# Step 3: Run predictions
print("🔮 Testing model predictions...")
os.system("python src/predict.py")

print("✅ CI/CD pipeline completed successfully!")
