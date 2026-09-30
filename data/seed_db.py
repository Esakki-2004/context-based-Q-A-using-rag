"""
Seed script — creates a sample SQLite database for testing the system.
Run once: python data/seed_db.py
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "database.db")


def seed():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.executescript("""
    CREATE TABLE IF NOT EXISTS departments (
        id      INTEGER PRIMARY KEY,
        name    TEXT NOT NULL,
        budget  REAL NOT NULL,
        head    TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS employees (
        id            INTEGER PRIMARY KEY,
        name          TEXT NOT NULL,
        department_id INTEGER NOT NULL REFERENCES departments(id),
        role          TEXT NOT NULL,
        salary        REAL NOT NULL,
        hire_date     TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS projects (
        id            INTEGER PRIMARY KEY,
        title         TEXT NOT NULL,
        department_id INTEGER NOT NULL REFERENCES departments(id),
        budget        REAL NOT NULL,
        status        TEXT NOT NULL CHECK(status IN ('active','completed','on_hold'))
    );

    DELETE FROM projects;
    DELETE FROM employees;
    DELETE FROM departments;

    INSERT INTO departments VALUES
        (1, 'Engineering',  2500000, 'Alice Sharma'),
        (2, 'Marketing',    800000,  'Bob Chen'),
        (3, 'HR',           400000,  'Carol White'),
        (4, 'Data Science', 1200000, 'Diana Patel');

    INSERT INTO employees VALUES
        (1,  'Alice Sharma',  1, 'VP Engineering',    145000, '2019-03-01'),
        (2,  'James Liu',     1, 'Senior Engineer',   110000, '2020-07-15'),
        (3,  'Priya Nair',    1, 'Engineer',           85000, '2022-01-10'),
        (4,  'Bob Chen',      2, 'VP Marketing',      130000, '2018-11-20'),
        (5,  'Sara Kim',      2, 'Marketing Manager', 95000,  '2021-05-05'),
        (6,  'Carol White',   3, 'HR Director',       105000, '2017-08-12'),
        (7,  'Tom Nguyen',    3, 'HR Specialist',      70000, '2023-02-01'),
        (8,  'Diana Patel',   4, 'Head of Data Sci',  140000, '2020-01-15'),
        (9,  'Rohan Mehta',   4, 'Data Scientist',     98000, '2021-09-20'),
        (10, 'Aiko Tanaka',   4, 'ML Engineer',       105000, '2022-06-01');

    INSERT INTO projects VALUES
        (1, 'Platform Rewrite',     1, 600000,  'active'),
        (2, 'Q3 Ad Campaign',       2, 150000,  'completed'),
        (3, 'Employee Wellbeing',   3,  50000,  'active'),
        (4, 'Churn Prediction Model',4, 200000, 'active'),
        (5, 'API Gateway v2',       1, 300000,  'on_hold'),
        (6, 'Brand Refresh',        2,  80000,  'completed');
    """)

    conn.commit()
    conn.close()
    print(f"Database seeded at: {DB_PATH}")


if __name__ == "__main__":
    seed()
