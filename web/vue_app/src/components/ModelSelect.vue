<template>
  <q-select v-model="experiment" :options="experiments" label="Experiment" @update:model-value="getRuns"/>
  <q-select v-model="run" :options="runs" label="Run" v-if="experiment !== null" @update:model-value="getCheckpoints" />
  <q-select v-model="checkpoint" :options="checkpoints" label="Checkpoint" v-if="run !== null" @update:model-value="getResponses" />
  <q-select v-model="response" :options="responses" label="Response" v-if="checkpoint !== null" @update:model-value="commitSelection" />
</template>

<script>
import { defineComponent } from "vue";
import { createNamespacedHelpers } from "vuex";
const { mapState, mapActions } = createNamespacedHelpers("auctions");

let DEFAULT_DATA = {
  experiment: null,
  run: null,
  checkpoint: null,
  response: null,
}

export default defineComponent({
  name: "ModelSelect",
  data() {
    return { ...DEFAULT_DATA }
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
  },
  mounted() {
    this.getExperiments();
  },
  computed: mapState({
    experiments: function(state) { 
      let modelSelector = state.modelSelector[this.player];
      return modelSelector ? modelSelector.experiments.map(e => ({ label: e.name, value: e.pk })) : [];
    },
    runs: function(state) {
      let modelSelector = state.modelSelector[this.player];
      return modelSelector ? modelSelector.runs.map(e => ({ label: e.name, value: e.pk })) : [];
    },
    checkpoints: function(state) {
      let modelSelector = state.modelSelector[this.player];
      return modelSelector ? modelSelector.checkpoints.map(e => ({ label: e.t + (e.best ? ' (BEST)' : ''), value: e.pk })) : [];
    },
    responses: function(state) {
      let modelSelector = state.modelSelector[this.player];
      return modelSelector ? [{label: 'Model', value: null}].concat(modelSelector.responses.map(e => ({ label: e.name, value: e.pk }))) : [];
    }, 
  }),
  watch: {
    game(new_game, old_game) {
      this.clear();
      this.getExperiments();
    }
  },
  methods: {
    getExperiments: function() {
      let gamePk = this.game.id;
      let player = this.player;
      this.GET_GAME_EXPERIMENTS({ gamePk, player }).then(data => {
        this.experiment = this.experiments[0];
        this.run = null;
        this.checkpoint = null;
        this.response = null;
        this.getRuns();
      });
    },
    getRuns: function () {
      this.GET_GAME_RUNS({ gamePk: this.game.id, experimentPk: this.experiment.value, player: this.player }).then(data => {
        this.run = this.runs[0];
        this.checkpoint = null;
        this.response = null;
        this.getCheckpoints();
      });
    },
    getCheckpoints: function () {
      this.GET_RUN_CHECKPOINTS({ gamePk: this.game.id, runPk: this.run.value, player: this.player }).then(data => {
        this.checkpoint = this.checkpoints[0];
        this.response = null;
        this.getResponses();
      });
    },
    getResponses: function () {
      this.GET_CHECKPOINT_RESPONSES({ checkpointPk: this.checkpoint.value, player:this.player }).then(data => {
        this.response = this.responses[0];
        this.commitSelection();
      });
    },
    commitSelection: function() {
      let player = this.player;
      let game = this.game;
      this.$store.commit('auctions/SET_SELECTOR', { player, game, ...this.$data} );
    },
    clear: function() {
      Object.assign(this.$data, { ...DEFAULT_DATA })
    },
    ...mapActions(["GET_GAME_RUNS", "GET_RUN_CHECKPOINTS", "GET_CHECKPOINT_RESPONSES", "GET_GAME_EXPERIMENTS"]),
  },
});
</script>
