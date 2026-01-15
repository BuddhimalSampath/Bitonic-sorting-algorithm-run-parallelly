from flask import Flask, render_template, request, jsonify
import subprocess
import csv
import os

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/run", methods=["POST"])
def run_sort():
    core_list = request.json["cores"]
    results = []
    
    # Ensure results file exists or overwrite
    with open("results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cores", "time"])
        writer.writeheader()

    for cores in core_list:
        # VALIDATION: Check if power of 2
        if (cores & (cores - 1) != 0) and cores != 0:
            print(f"Skipping {cores} cores: Not a power of 2 (Required for Bitonic Sort)")
            continue

        print(f"Running on {cores} cores...")
        
        # Command: mpiexec -n <cores> python parallel_run_copy.py
        cmd = ["mpiexec", "-n", str(cores), "python", "parallel_run_copy.py"]

        try:
            # Run the MPI script and capture output
            process = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = process.stdout.strip()
            
            # Expecting output format: "CORES,TIME"
            # We split by comma in case there are other prints, we take the last line ideally
            lines = output.split('\n')
            last_line = lines[-1] 
            
            if "," in last_line:
                c, t = last_line.split(",")
                entry = {
                    "cores": int(c),
                    "time": float(t)
                }
                results.append(entry)
                
                # Append to CSV immediately
                with open("results.csv", "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["cores", "time"])
                    writer.writerow(entry)
            else:
                print(f"Unexpected output format: {output}")

        except subprocess.CalledProcessError as e:
            print(f"Error executing MPI on {cores} cores.")
            print("STDERR:", e.stderr)
        except Exception as e:
            print(f"General Error: {e}")

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)