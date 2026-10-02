
try:
    with open("evaluate_dqn_output.txt", "r", encoding="utf-16") as f:
        print(f.read())
except Exception as e:
    print(f"Error reading utf-16: {e}")
    try:
        with open("evaluate_dqn_output.txt", "r", encoding="utf-8") as f:
            print(f.read())
    except Exception as e2:
        print(f"Error reading utf-8: {e2}")
        print("Could not read file.")
