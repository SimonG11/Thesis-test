# unit_tests.py
import numpy as np
import logging
from metrics import update_archive_with_crowding
from rcpsp_model import RCPSPModel
from tasks import get_default_tasks


def run_unit_tests() -> None:
    # Test the archive update function.
    sol1 = np.array([1, 2, 3])
    obj1 = np.array([10, 20, 30])
    sol2 = np.array([2, 3, 4])
    obj2 = np.array([12, 22, 32])
    archive = []
    archive = update_archive_with_crowding(archive, (sol1, obj1))
    archive = update_archive_with_crowding(archive, (sol2, obj2))
    if len(archive) != 1:
        logging.error("Unit Test Failed: Archive contains dominated solutions.")
    else:
        logging.info("Unit Test Passed: Archive update produces non-dominated set.")
    
    # Test RCPSP model schedule computation.
    workers = {"Developer": 5, "Manager": 2, "Tester": 3}
    worker_cost = {"Developer": 50, "Manager": 75, "Tester": 40}
    tasks = get_default_tasks()
    model = RCPSPModel(tasks, workers, worker_cost)
    x = np.array([task["min"] for task in tasks])
    schedule, ms = model.compute_schedule(x)
    if schedule and ms > 0:
        logging.info("Unit Test Passed: RCPSP schedule computed successfully.")
    else:
        logging.error("Unit Test Failed: RCPSP schedule computation issue.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_unit_tests()
