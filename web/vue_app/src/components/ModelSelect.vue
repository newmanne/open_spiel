<template>
  <q-select
    v-model="experiment"
    :options="experiments"
    label="Experiment"
    @update:model-value="getRuns"
  />
  <q-select
    v-model="run"
    :options="runs"
    label="Run"
    v-if="experiment !== null"
    @update:model-value="getCheckpoints"
  />
  <q-select
    v-model="checkpoint"
    :options="checkpoints"
    label="Checkpoint"
    v-if="run !== null"
    @update:model-value="getResponses"
  />
  <q-select
    v-model="response"
    :options="responses"
    label="Response"
    v-if="checkpoint !== null"
    @update:model-value="commitSelection"
  />
</template>

<script>
import { defineComponent } from "vue";
import { createNamespacedHelpers } from "vuex";
const { mapState, mapActions } = createNamespacedHelpers("auctions");
import { FMT } from "../utils.js"

export default defineComponent({
  name: "ModelSelect",
  emits: ["updateSelection"],
  data() {
    return {
      experiment: null,
      run: null,
      checkpoint: null,
      response: null,
      experiments: [],
      runs: [],
      checkpoints: [],
      responses: [],
    };
  },
  props: {
    game: {
      type: Object,
      required: true,
    },
    player: {
      type: Number,
      required: true,
    },
    depth: {
      type: String,
      default: "response",
    },
  },
  mounted() {
    this.getExperiments();
  },
  computed: mapState({}),
  watch: {
    game(new_game, old_game) {
      this.clear();
      this.getExperiments();
    },
  },
  methods: {
    getExperiments: function () {
      let gamePk = this.game.id;
      let player = this.player;
      this.GET_GAME_EXPERIMENTS({ gamePk, player }).then((data) => {
        this.experiments = data.map((e) => ({ label: e.name, value: e.pk }));
        this.experiment = this.experiments[0];
        this.run = null;
        this.checkpoint = null;
        this.response = null;
        this.getRuns();
      });
    },
    getRuns: function () {
      this.GET_GAME_RUNS({
        gamePk: this.game.id,
        experimentPk: this.experiment.value,
        player: this.player,
      }).then((data) => {
        this.runs = data.map((e) => ({ label: e.name, value: e.pk }));
        this.runs.sort((a, b) => a.label.localeCompare(b.label));
        this.run = this.runs[0];
        this.checkpoint = null;
        this.response = null;
        this.getCheckpoints();
      });
    },
    getCheckpoints: function () {
      this.GET_RUN_CHECKPOINTS({
        gamePk: this.game.id,
        runPk: this.run.value,
        player: this.player,
      }).then((data) => {
        this.checkpoints = data.map((e) => ({
          label: FMT(e.t) + (e.best ? " (Lowest ApproxNashConv)" : ""),
          value: e.pk,
        }));
        this.checkpoint = this.checkpoints[0];
        this.response = null;
        this.getResponses();
      });
    },
    getResponses: function () {
      this.GET_CHECKPOINT_RESPONSES({
        checkpointPk: this.checkpoint.value,
        player: this.player,
      }).then((data) => {
        this.responses = [{ label: "Model", value: null }].concat(
          data.map((e) => ({ label: e.name, value: e.pk }))
        );
        this.response = this.responses[0];
        this.commitSelection();
      });
    },
    commitSelection: function () {
      let data = {
        player: this.player,
        checkpoint: this.checkpoint?.value,
        response: this.response?.value,
        experiment: this.experiment?.value,
        run: this.run?.value
      }
      this.$emit("updateSelection", data);
    },
    clear: function () {
      Object.assign(this.$data, this.$options.data());
    },
    ...mapActions([
      "GET_GAME_RUNS",
      "GET_RUN_CHECKPOINTS",
      "GET_CHECKPOINT_RESPONSES",
      "GET_GAME_EXPERIMENTS",
    ]),
  },
});
</script>
