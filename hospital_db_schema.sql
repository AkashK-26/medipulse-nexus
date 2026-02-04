-- Hospital Operations Analytics Database Schema

-- 1. Core Tables
CREATE TABLE IF NOT EXISTS branches (
    branch_id SERIAL PRIMARY KEY,
    branch_name VARCHAR(100) NOT NULL,
    location VARCHAR(100),
    capacity_beds INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS departments (
    dept_id SERIAL PRIMARY KEY,
    dept_code VARCHAR(20) UNIQUE NOT NULL,
    dept_name VARCHAR(50) NOT NULL,
    branch_id INTEGER REFERENCES branches(branch_id),
    bed_count INTEGER,
    icu_beds INTEGER,
    ventilators INTEGER
);

CREATE TABLE IF NOT EXISTS doctors (
    doctor_id SERIAL PRIMARY KEY,
    doctor_name VARCHAR(100),
    dept_id INTEGER REFERENCES departments(dept_id),
    specialization VARCHAR(50),
    shift_start TIME,
    shift_end TIME,
    max_patients_per_day INTEGER
);

CREATE TABLE IF NOT EXISTS patients (
    patient_id SERIAL PRIMARY KEY,
    age INTEGER,
    gender VARCHAR(10),
    insurance_type VARCHAR(50),
    admission_type VARCHAR(20),
    branch_id INTEGER REFERENCES branches(branch_id)
);

CREATE TABLE IF NOT EXISTS admissions (
    admission_id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(patient_id),
    dept_id INTEGER REFERENCES departments(dept_id),
    doctor_id INTEGER REFERENCES doctors(doctor_id),
    admission_datetime TIMESTAMP NOT NULL,
    discharge_datetime TIMESTAMP,
    diagnosis_category VARCHAR(50),
    procedure_code VARCHAR(20),
    procedure_name VARCHAR(100),
    outcome VARCHAR(20),
    total_cost DECIMAL(10,2),
    billing_breakdown JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS bed_occupancy (
    occupancy_id SERIAL PRIMARY KEY,
    dept_id INTEGER REFERENCES departments(dept_id),
    date_hour TIMESTAMP,
    occupied_beds INTEGER,
    available_beds INTEGER,
    icu_occupied INTEGER,
    ventilators_in_use INTEGER
);

CREATE TABLE IF NOT EXISTS staff_schedules (
    schedule_id SERIAL PRIMARY KEY,
    doctor_id INTEGER REFERENCES doctors(doctor_id),
    work_date DATE,
    hours_booked DECIMAL(4,2),
    hours_available DECIMAL(4,2),
    patient_count INTEGER
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_admissions_dates ON admissions(admission_datetime, discharge_datetime);
CREATE INDEX IF NOT EXISTS idx_admissions_dept ON admissions(dept_id);
CREATE INDEX IF NOT EXISTS idx_bed_occupancy_date ON bed_occupancy(date_hour);
CREATE INDEX IF NOT EXISTS idx_patients_branch ON patients(branch_id);
