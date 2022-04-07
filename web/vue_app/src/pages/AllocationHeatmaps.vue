<template>
  <q-page class="q-px-md row flex flex-center">
    <div class="col-9">
      <div class="text-h5 q-py-md">Allocation Heatmaps</div>
      <div class="q-pa-md shadow-box shadow-5">
        <div class="q-pb-md">
          <game-select @input="onGameSelected" />
        </div>
        <template v-if="game !== null">
          <span><b>Select run:</b></span>
          <model-select
            :game="game"
            depth="checkpoint"
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
          <template v-if="allocations !== null">
            <div
              v-for="(value, product) in allocations"
              :key="product"
              class="col-3"
            >
              <AllocationsHeatmap
                :service-area="product"
                :alloc="value"
                :show-zero="showZero"
                :bidders="bidders"
              />
            </div>
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
import AllocationsHeatmap from "../components/charts/AllocationHeatmap.vue";

export default defineComponent({
  name: "PageAllocationHeatmaps",
  components: {
    GameSelect,
    ModelSelect,
    AllocationsHeatmap,
  },
  mounted() {},
  data() {
    return {
      game: null,
      allocations: null,
      selector: {},
      showZero: false,
    };
  },
  computed: {
    products() {
      if (this.game === null) {
        return [];
      }
      return _.range(this.game.num_products);
    },
    bidders() {
      return _.range(this.game.num_players);
    },
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
      let checkpointPk = this.selector.checkpoint;
      this.GET_ALLOCATIONS({ checkpointPk }).then((data) => {
        this.allocations = data.allocations;
      });
    },
    ...mapActions(["GET_ALLOCATIONS"]),
  },
});
</script>
