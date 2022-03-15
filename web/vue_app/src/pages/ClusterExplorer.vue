<template>
  <q-page class="q-px-md row flex flex-center">
    <div class="col-9">
      <div class="text-h5 q-py-md">Cluster Explorer</div>
      <span>First, use opening explorer. Then come here.</span>
      <div class="q-pa-md shadow-box shadow-5">
        <div class="text-h6">Cluster Graphs</div>

        <template v-if="dropdown_players.length > 0">
          <div>
            <q-select
              label="Player"
              v-model="selected_player"
              :options="dropdown_players"
              emit-value
            />
            <template v-if="selected_player !== null">
              <div>
                <b>Player {{ selected_player }} model:</b>
              </div>
              <iframe
                :srcdoc="clusters_bokeh[selected_player]"
                style="min-width: 1000px; min-height: 600px"
              ></iframe>
            </template>
          </div>
        </template>
        <template v-else>
          <span>Load using the opening explorer</span>
        </template>
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

export default defineComponent({
  name: "PageClusterExplorer",
  components: {},
  mounted() {},
  data() {
    return {
      selected_player: null,
    };
  },
  computed: {
    dropdown_players() {
      return Object.keys(this.clusters_bokeh).map(s => parseInt(s));
    },
    ...mapState({
      clusters_bokeh: (state) =>
        state.samples.clusters_bokeh ? state.samples.clusters_bokeh : {},
    }),
  },
  methods: {
    ...mapActions([]),
  },
});
</script>
