"""
Run the AI agent validation suite from the command line (same as the button on the AI Agent page).

    python run_agent_validation.py            # all golden scenarios
    python run_agent_validation.py S03 S09    # selected scenarios

Results are saved to the database and to data/agent_validation_results.csv, which the dashboard shows when the
database has no run yet (for example on Streamlit Cloud).
"""
import sys

import agent_validation as AV
import database as db
import matching as M
import performance as P


def main(scenario_ids=None):
    db.init_db()
    master = M.prepare_master(db.load_master().to_dict("records"))
    results = db.load_monitoring_results()
    if results.empty:
        results = P.run_monitoring(master)
        db.save_monitoring_results(results)
        results = db.load_monitoring_results()
    runs = AV.run_validation(results, master, scenario_ids=scenario_ids,
                             progress=lambda n, k, d: print(f"[{n}/{k}] {d}", flush=True))
    db.save_agent_validation_runs(runs)
    table, outcome = AV.summarise(runs)
    print(table.to_string(index=False))
    print("Outcome (before expert review):", outcome)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
