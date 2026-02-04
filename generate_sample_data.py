import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import random
import json

# Database connection
engine = create_engine('postgresql://hospital_user:secure_password@localhost:5432/hospital_analytics')

def get_specialization(dept_name):
    """Map department name to doctor specialization"""
    mapping = {
        'Cardiology': 'Cardiologist',
        'Oncology': 'Oncologist', 
        'Orthopedics': 'Orthopedic Surgeon',
        'Pediatrics': 'Pediatrician',
        'Emergency': 'Emergency Physician',
        'General Medicine': 'General Physician'
    }
    return mapping.get(dept_name, 'General Physician')

def generate_sample_data():
    """Generate realistic hospital data for testing"""
    print("🏥 Generating sample hospital data...")
    
    # 1. Branches
    print("Creating branches...")
    branches = pd.DataFrame({
        'branch_name': ['Mumbai Central', 'Delhi North', 'Bangalore South', 'Chennai East'],
        'location': ['Mumbai', 'Delhi', 'Bangalore', 'Chennai'],
        'capacity_beds': [500, 400, 350, 300]
    })
    branches.to_sql('branches', engine, if_exists='append', index=False)
    
    branch_ids = pd.read_sql("SELECT branch_id FROM branches", engine)['branch_id'].tolist()
    
    # 2. Departments
    print("Creating departments...")
    depts = []
    dept_configs = {
        'Cardiology': (50, 10, 5),
        'Oncology': (40, 8, 4),
        'Orthopedics': (60, 5, 3),
        'Pediatrics': (80, 15, 6),
        'Emergency': (30, 20, 10),
        'General Medicine': (100, 12, 8)
    }
    
    for branch_id in branch_ids:
        for dept_name, (beds, icu, vents) in dept_configs.items():
            depts.append({
                'dept_code': f"{dept_name[:3].upper()}{branch_id}",
                'dept_name': dept_name,
                'branch_id': branch_id,
                'bed_count': beds,
                'icu_beds': icu,
                'ventilators': vents
            })
    
    depts_df = pd.DataFrame(depts)
    depts_df.to_sql('departments', engine, if_exists='append', index=False)
    
    dept_data = pd.read_sql("SELECT dept_id, dept_name, bed_count, icu_beds, ventilators FROM departments", engine)
    
    # 3. Doctors
    print("Creating doctors...")
    doctors = []
    first_names = ['Rajesh', 'Priya', 'Amit', 'Sunita', 'Vikram', 'Anita', 'Suresh', 'Deepa', 'Rahul', 'Neha']
    last_names = ['Sharma', 'Patel', 'Kumar', 'Singh', 'Gupta', 'Reddy', 'Nair', 'Desai', 'Iyer', 'Mehta']
    
    for _, dept in dept_data.iterrows():
        num_doctors = random.randint(5, 12)
        for i in range(num_doctors):
            doctors.append({
                'doctor_name': f"Dr. {random.choice(first_names)} {random.choice(last_names)}",
                'dept_id': dept['dept_id'],
                'specialization': get_specialization(dept['dept_name']),
                'shift_start': f"{random.randint(8, 10)}:00:00",
                'shift_end': f"{random.randint(17, 21)}:00:00",
                'max_patients_per_day': random.randint(15, 30)
            })
    
    doctors_df = pd.DataFrame(doctors)
    doctors_df.to_sql('doctors', engine, if_exists='append', index=False)
    
    doctor_data = pd.read_sql("SELECT doctor_id, dept_id FROM doctors", engine)
    
    # 4. Patients and Admissions
    print("Creating patients and admissions (this may take a minute)...")
    patients = []
    admissions = []
    
    insurance_types = ['Cash', 'Private Insurance', 'Government Scheme', 'Corporate']
    admission_types = ['Emergency', 'Scheduled']
    diagnoses = ['Heart Failure', 'Cancer', 'Fracture', 'Pneumonia', 'Accident', 'Diabetes', 'Hypertension', 'Stroke', 'Appendicitis', 'Asthma']
    outcomes = ['Recovered', 'Improved', 'Transferred', 'Deceased']
    procedures = ['Surgery', 'Chemotherapy', 'Consultation', 'Imaging', 'Lab Tests', 'Therapy', 'Angioplasty', 'Dialysis']
    
    start_date = datetime.now() - timedelta(days=90)
    
    # Generate 2000 patients
    for i in range(2000):
        branch_id = random.choice(branch_ids)
        patients.append({
            'age': random.randint(1, 85),
            'gender': random.choice(['Male', 'Female']),
            'insurance_type': random.choice(insurance_types),
            'admission_type': random.choice(admission_types),
            'branch_id': branch_id
        })
    
    patients_df = pd.DataFrame(patients)
    patients_df.to_sql('patients', engine, if_exists='append', index=False)
    patient_ids = pd.read_sql("SELECT patient_id FROM patients", engine)['patient_id'].tolist()
    
    # Generate admissions
    for patient_id in patient_ids:
        for _ in range(random.randint(1, 3)):
            dept = dept_data.sample(1).iloc[0]
            dept_doctors = doctor_data[doctor_data['dept_id'] == dept['dept_id']]
            if dept_doctors.empty:
                continue
            
            doctor_id = dept_doctors.sample(1).iloc[0]['doctor_id']
            
            admission_time = start_date + timedelta(
                days=random.randint(0, 90),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            # Length of stay varies by department
            if dept['dept_name'] == 'Emergency':
                los_days = random.uniform(0.5, 3)
            elif dept['dept_name'] == 'Cardiology':
                los_days = random.uniform(3, 10)
            elif dept['dept_name'] == 'Oncology':
                los_days = random.uniform(5, 21)
            else:
                los_days = random.uniform(2, 7)
            
            discharge_time = admission_time + timedelta(days=los_days) if random.random() > 0.1 else None
            
            base_cost = los_days * 5000
            procedure_cost = random.randint(10000, 100000) if random.random() > 0.3 else 0
            total_cost = base_cost + procedure_cost
            
            outcome = random.choice(outcomes)
            if discharge_time is None:
                outcome = None
            
            admissions.append({
                'patient_id': patient_id,
                'dept_id': dept['dept_id'],
                'doctor_id': doctor_id,
                'admission_datetime': admission_time,
                'discharge_datetime': discharge_time,
                'diagnosis_category': random.choice(diagnoses),
                'procedure_code': f"PROC{random.randint(100, 999)}",
                'procedure_name': random.choice(procedures),
                'outcome': outcome,
                'total_cost': total_cost,
                'billing_breakdown': json.dumps({
                    'room_charges': base_cost * 0.4,
                    'doctor_fees': base_cost * 0.2,
                    'medicines': base_cost * 0.15,
                    'procedures': procedure_cost,
                    'other': base_cost * 0.25
                })
            })
    
    admissions_df = pd.DataFrame(admissions)
    admissions_df.to_sql('admissions', engine, if_exists='append', index=False)
    
    # Generate bed occupancy data
    print("Generating bed occupancy data...")
    bed_occupancy = []
    
    for single_date in (datetime.now() - timedelta(days=n) for n in range(30)):
        for hour in range(24):
            timestamp = single_date + timedelta(hours=hour)
            
            for _, dept in dept_data.iterrows():
                base_occupancy = 0.7
                hour_factor = 1.2 if 9 <= hour <= 17 else 0.8
                random_factor = random.uniform(0.8, 1.2)
                
                occupied = int(dept['bed_count'] * base_occupancy * hour_factor * random_factor)
                icu_occupied = int(dept['icu_beds'] * random.uniform(0.6, 0.9))
                vent_used = int(dept['ventilators'] * random.uniform(0.3, 0.7))
                
                bed_occupancy.append({
                    'dept_id': dept['dept_id'],
                    'date_hour': timestamp,
                    'occupied_beds': min(occupied, dept['bed_count']),
                    'available_beds': dept['bed_count'] - min(occupied, dept['bed_count']),
                    'icu_occupied': min(icu_occupied, dept['icu_beds']),
                    'ventilators_in_use': min(vent_used, dept['ventilators'])
                })
    
    occ_df = pd.DataFrame(bed_occupancy)
    occ_df.to_sql('bed_occupancy', engine, if_exists='append', index=False, chunksize=1000)
    
    # Generate staff schedules
    print("Generating staff schedules...")
    staff_schedules = []
    
    for single_date in (datetime.now() - timedelta(days=n) for n in range(30)):
        for _, doctor in doctor_data.iterrows():
            hours_available = random.uniform(7, 10)
            utilization = random.uniform(0.6, 0.95)
            hours_booked = hours_available * utilization
            patient_count = int(hours_booked * random.uniform(2, 4))
            
            staff_schedules.append({
                'doctor_id': doctor['doctor_id'],
                'work_date': single_date.date(),
                'hours_booked': round(hours_booked, 2),
                'hours_available': round(hours_available, 2),
                'patient_count': patient_count
            })
    
    sched_df = pd.DataFrame(staff_schedules)
    sched_df.to_sql('staff_schedules', engine, if_exists='append', index=False)
    
    print("✅ Sample data generation complete!")
    print(f"   - {len(branches)} branches")
    print(f"   - {len(depts)} departments")
    print(f"   - {len(doctors)} doctors")
    print(f"   - {len(patients)} patients")
    print(f"   - {len(admissions)} admissions")
    print(f"   - {len(bed_occupancy)} occupancy records")

if __name__ == "__main__":
    generate_sample_data()

