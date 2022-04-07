<template>
  <q-page
    class="text-center q-pa-md flex flex-center"
  >
    <q-select
      :filter="filterBySubstring"
      autofocus-filter="true"
      v-model="selectedExperiment"
      :options="experiments"
      label="Select an experiment"
      @update:model-value="getRuns"
    ></q-select>
    <br />
    <q-select
      v-if="selectedExperiment"
      :filter="filterBySubstring"
      autofocus-filter="true"
      v-model="selectedRun"
      :options="runs"
      label="Select a run"
      @update:model-value="getCheckpoints"
    ></q-select>
    <br />
    <q-select
      v-if="selectedRun"
      :filter="filterBySubstring"
      autofocus-filter="true"
      v-model="selectedCheckpoint"
      :options="checkpoints"
      label="Select a checkpoint"
      @update:model-value="checkpointSelected"
    ></q-select>
  </q-page>
</template>

<script>
import { defineComponent } from "vue";
import { createNamespacedHelpers } from "vuex";
const { mapState, mapActions } = createNamespacedHelpers("auctions");

export default defineComponent({
  name: "PageIndex",
  data() {
    return {
      selectedExperiment: null,
      selectedRun: null,
      selectedCheckpoint: null,
    };
  },
  mounted() {
    // this.GET_EXPERIMENTS();
  },
  computed: mapState({
    experiments: (state) =>
      state.experiments.map((e) => ({ label: e.name, value: e.pk })),
    runs: (state) => state.runs.map((e) => ({ label: e.name, value: e.pk })),
    checkpoints: (state) =>
      state.checkpoints.map((e) => ({ label: e.t, value: e.pk })),
  }),
  methods: {
    filterBySubstring(s, obj) {
      return obj.label.toLowerCase().indexOf(s.toLowerCase()) !== -1;
    },
    getRuns: function () {
      let experimentPk = this.selectedExperiment.value;
      this.GET_RUNS({ experimentPk });
    },
    getCheckpoints: function () {
      let runPk = this.selectedRun.value;
      this.GET_CHECKPOINTS({ runPk });
    },
    checkpointSelected: function () {
      console.log("HERE");
    },
    ...mapActions(["GET_EXPERIMENTS", "GET_RUNS", "GET_CHECKPOINTS"]),
  },
});
</script>
