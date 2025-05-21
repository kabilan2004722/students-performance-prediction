from flask import Flask, render_template, request, redirect, url_for, send_file
import sqlite3
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
DATABASE = 'students.db'

def init_db():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS students (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          student_id TEXT,
                          name TEXT,
                          subject TEXT,
                          cia1 INTEGER,
                          cia2 INTEGER,
                          model_exam INTEGER,
                          predicted_mark REAL)''')
        conn.commit()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    num_students = int(request.form['num_students'])
    students = []
    
    for i in range(num_students):
        student = {
            'student_id': request.form[f'student_id_{i}'],
            'name': request.form[f'name_{i}'],
            'subject': request.form[f'subject_{i}'],
            'cia1': int(request.form[f'cia1_{i}']),
            'cia2': int(request.form[f'cia2_{i}']),
            'model_exam': int(request.form[f'model_exam_{i}'])
        }
        students.append(student)
    
    df = pd.DataFrame(students)
    features = df[['cia1', 'cia2', 'model_exam']]
    model = xgb.XGBRegressor()
    model.fit(features, features.mean(axis=1))
    df['predicted_mark'] = model.predict(features)
    
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        for _, row in df.iterrows():
            cursor.execute("INSERT INTO students (student_id, name, subject, cia1, cia2, model_exam, predicted_mark) VALUES (?, ?, ?, ?, ?, ?, ?)", 
                           (row['student_id'], row['name'], row['subject'], row['cia1'], row['cia2'], row['model_exam'], row['predicted_mark']))
        conn.commit()
    
    # Generate Pie Chart for Performance Categories
    categories = ['Good (>40)', 'Average (>30)', 'Low (>20)', 'Fail (<20)']
    counts = [sum(df['predicted_mark'] > 40),
              sum((df['predicted_mark'] > 30) & (df['predicted_mark'] <= 40)),
              sum((df['predicted_mark'] > 20) & (df['predicted_mark'] <= 30)),
              sum(df['predicted_mark'] <= 20)]
    
    plt.figure(figsize=(6,6))
    plt.pie(counts, labels=categories, autopct='%1.1f%%', colors=['green', 'yellow', 'orange', 'red'])
    plt.title('Student Performance Distribution')
    pie_chart_path = os.path.join('static', 'performance_pie_chart.png')
    plt.savefig(pie_chart_path)
    plt.close()
    
    return render_template('result.html', students=df.to_dict(orient='records'), pie_chart_url=pie_chart_path)

@app.route('/view')
def view():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students")
        students = cursor.fetchall()
    return render_template('view.html', students=students)

@app.route('/delete/<int:id>')
def delete(id):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM students WHERE id=?", (id,))
        conn.commit()
    return redirect(url_for('view'))

@app.route('/download')
def download():
    with sqlite3.connect(DATABASE) as conn:
        df = pd.read_sql_query("SELECT * FROM students", conn)
    file_path = 'static/student_records.xlsx'
    df.to_excel(file_path, index=False)
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
