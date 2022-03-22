<template>
  <q-page class="q-px-md row flex flex-center">
    <div class="col-9">
      <div class="text-h5 q-py-md">Trajectory Plots</div>
      <div class="q-pa-md shadow-box shadow-5">
        <div class="q-pb-md">
          <game-select @input="onGameSelected" />
        </div>
        <template v-if="game !== null">
          <span><b>Select run:</b></span>
          <model-select
            :game="game"
            depth="run"
            @updateSelection="onSelectionUpdated"
          />
        </template>
        <q-btn
          label="Run"
          icon="model_training"
          color="primary"
          @click="getPlot()"
        />
        <div>
          <template v-if="plot !== null">
            <iframe
              :srcdoc="plot"
              style="min-width: 1000px; min-height: 600px"
            ></iframe>
          </template>
        </div>
      </div>
    </div>
  </q-page>
</template>

<script>
import { defineComponent } from "vue";
import { createNamespacedHelpers } from "vuex";
const { mapState, mapActions } = createNamespacedHelpers("auctions");
import _ from "lodash";
import { FMT_STR, FMT } from "../utils.js";
import GameSelect from "../components/GameSelect.vue";
import ModelSelect from "../components/ModelSelect.vue";

export default defineComponent({
  name: "PageTrajecotryExplorer",
  components: {
    GameSelect,
    ModelSelect,
  },
  mounted() {},
  data() {
    return {
      game: null,
      plot: null,
      selector: {},
    };
  },
  computed: {
    ...mapState({}),
  },
  methods: {
    onGameSelected(game) {
      this.game = game.value;
    },
    onSelectionUpdated(evt) {
      this.selector = evt;
    },
    getPlot() {
      let runPk = this.selector.run;
      this.GET_TRAJECTORY_PLOT({ runPk }).then((data) => {
        this.plot = data.bokeh_js;
      });
    },
    ...mapActions(["GET_TRAJECTORY_PLOT"]),
  },
});
</script>
