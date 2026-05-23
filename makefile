IMAGE     := bobsim
DOCKER_RUN := docker run --rm -v "$(CURDIR):/workspace" -e PYTHONPATH=/workspace $(IMAGE)

.PHONY: init setup rebuild \
        shell shell-doe \
        sim-doe sim-steady-state sim-transient sim-four-post \
        clean-doe clean-standard clean

# ── Setup ──────────────────────────────────────────────────────────────────

init:
	git submodule update --init --recursive

setup:
	docker build -t $(IMAGE) .

rebuild:
	docker build --no-cache -t $(IMAGE) .

# ── Shells ─────────────────────────────────────────────────────────────────

shell:
	docker run --rm -it -v "$(CURDIR):/workspace" -e PYTHONPATH=/workspace $(IMAGE) bash

shell-doe:
	docker run --rm -it -v "$(CURDIR):/workspace" -e PYTHONPATH=/workspace -w /workspace/_4_OptSim $(IMAGE) bash

# ── Sims ───────────────────────────────────────────────────────────────────
# All sims run fully inside Docker — no local OMC or Python install needed.
# Build is skipped if the executable already exists; run make clean-standard to force rebuild.

sim-doe:
	docker run --rm -v "$(CURDIR):/workspace" -e PYTHONPATH=/workspace -w /workspace/_4_OptSim $(IMAGE) python run_doe.py

sim-steady-state:
	$(DOCKER_RUN) bash -c "\
		if [ ! -f _3_StandardSim/Build/VehicleSim/BobLib.Standards.VehicleSim ]; then \
		    omc /workspace/_3_StandardSim/build_vehicle_sim.mos && \
		    [ -f _3_StandardSim/Build/VehicleSim/BobLib.Standards.VehicleSim ] || \
		        { echo 'ERROR: OMC build failed - executable not produced'; exit 1; }; \
		fi && \
		python -m _3_StandardSim.SteadyStateEval.steady_state_eval_sim"

sim-transient:
	$(DOCKER_RUN) bash -c "\
		if [ ! -f _3_StandardSim/Build/VehicleSim/BobLib.Standards.VehicleSim ]; then \
		    omc /workspace/_3_StandardSim/build_vehicle_sim.mos && \
		    [ -f _3_StandardSim/Build/VehicleSim/BobLib.Standards.VehicleSim ] || \
		        { echo 'ERROR: OMC build failed - executable not produced'; exit 1; }; \
		fi && \
		python -m _3_StandardSim.TransientEval.transient_eval_sim"

sim-four-post:
	$(DOCKER_RUN) bash -c "\
		if [ ! -f _3_StandardSim/Build/FourPostSim/BobLib.Standards.FourPostSim ]; then \
		    omc /workspace/_3_StandardSim/build_four_post_sim.mos && \
		    [ -f _3_StandardSim/Build/FourPostSim/BobLib.Standards.FourPostSim ] || \
		        { echo 'ERROR: OMC build failed - executable not produced'; exit 1; }; \
		fi && \
		python -m _3_StandardSim.FourPostEval.four_post_eval_sim"

# ── Clean ──────────────────────────────────────────────────────────────────
# compile_error_*.log files are preserved for debugging.

clean-doe:
	$(DOCKER_RUN) bash -c "\
		find /workspace/_4_OptSim/population -mindepth 1 ! -name '.gitkeep' -delete && \
		find /workspace/_4_OptSim/results    -mindepth 1 ! -name '.gitkeep' -delete"

clean-standard:
	$(DOCKER_RUN) bash -c "\
		find /workspace/_3_StandardSim/Build   -mindepth 1 ! -name '.gitkeep' -delete ; \
		find /workspace/_3_StandardSim/results -mindepth 1 ! -name '.gitkeep' -delete"

clean:
	$(DOCKER_RUN) bash -c "\
		find /workspace -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null; \
		find /workspace -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete 2>/dev/null; \
		rm -rf /workspace/.pytest_cache /workspace/.mypy_cache /workspace/.ruff_cache; \
		rm -rf /workspace/.coverage /workspace/htmlcov; \
		rm -rf /workspace/build /workspace/dist; \
		find /workspace -maxdepth 2 -name '*.egg-info' -exec rm -rf {} + 2>/dev/null; \
		echo 'Clean complete'"
