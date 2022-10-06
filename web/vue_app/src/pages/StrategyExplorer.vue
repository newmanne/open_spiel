<template>
  <q-page class="q-px-md row flex flex-center">
    <div class="col-9">
      <div class="text-h5 q-py-md">Strategy Explorer</div>
      <div class="q-pa-md shadow-box shadow-5">
        <div class="text-h6">Settings</div>
        <span><b>Models</b></span>
        <div class="q-pb-md">
          <game-select @input="onGameSelected"/>
        </div>
        <template v-if="game !== null">
          <div class="q-py-md" v-for="player in players" :key="player">
            <span><b>Player {{player}} model:</b></span>
            <model-select :game="game" :player="player" @updateSelection="onSelectionUpdated"/>
          </div>
        </template>
        <span><b>Sampler settings</b></span>
        <sampler-settings @input="onSampleParamsChanged"/>
        <q-btn
          label="Run"
          icon="model_training"
          color="primary"
          :disabled="!readyForSamples"
          :loading="loading_samples"
          @click="onClickRun()"
        />

      </div>

      <br/>
      <div class="q-pa-md shadow-box shadow-5">
        <div class="text-h6">Game Tree</div>
        <template v-if="!selected_samples">
          Select models and press Run to display game tree.
        </template>
        <template v-else>
          <template v-if="previous_nodes.length > 0">
            <span><b>History</b></span>
            <q-table
            :rows="previous_nodes"
            :columns="previous_columns"
            :visible-columns="previous_visible_columns"
            row-key="action"
            table-style="white-space: pre;"
            hide-bottom
            :hide-pagination="true"
            :rows-per-page-options="[0]"
            @row-click="onClickPreviousActionRow"
            />
            <br />
          </template>
          <template v-if="loading_samples">
            Sampling game trees...
          </template>
          <template v-else>
              <template v-if="next_nodes.length > 0">
                <span><b>Next states/actions</b></span>
                <q-table
                  :rows="next_nodes"
                  :columns="next_columns"
                  :visible-columns="next_visible_columns"
                  :pagination="pagination"
                  row-key="action"
                  table-style="white-space: pre;"
                  hide-bottom
                  :hide-pagination="true"
                  :rows-per-page-options="[0]"
                  @row-click="onClickNextActionRow"
                />
              </template>
            </template>
          </template>
      </div>
    </div>
  </q-page>
</template>

<script>
import { defineComponent } from "vue";
import { createNamespacedHelpers } from "vuex";
const { mapState, mapActions } = createNamespacedHelpers("auctions");
import GameSelect from "../components/GameSelect.vue";
import ModelSelect from "../components/ModelSelect.vue";
import SamplerSettings from '../components/SamplerSettings.vue'
import _ from 'lodash' 
import { FMT_STR, FMT } from "../utils.js";

function formatNode(outcomes) {
  // format arrays as strings
  Object.keys(outcomes).forEach(k => (
    outcomes[k] = Array.isArray(outcomes[k]) ? JSON.stringify(outcomes[k], null, ' ') : outcomes[k]
  ))
  return outcomes;
}

const HIDDEN_COLUMNS = ["depth", "action", "type", "max_cp"]
const COLUMN_FORMATS = {
  pretty_str: {
    label: 'Action',
    priority: 0,
    format: (val, row) => val,
  },
  num_plays: {
    priority: 1,
  },
  pct_plays: {
    label: '% Played',
    priority: 2,
  },
  straightforward_clock_profit: {
    classes: row => row.max_cp ? 'bg-amber-5' : '',
  }
};

function columnifier(fields, sortable) {
    let columns = _.map(fields, c => {
      if (COLUMN_FORMATS.hasOwnProperty(c)) {
        return {name: c, sortable: sortable, ...COLUMN_FORMATS[c]}
      } else {
        return {
          name: c, sortable: sortable,
        }
      }
    });
    columns = _.map(columns, column => {
      if (!column.hasOwnProperty('label')) {
        column.label = FMT_STR(column.name);
      }
      if (!column.hasOwnProperty('field')) {
        column.field = column.name;
      }
      if (!column.hasOwnProperty('priority')) {
        column.priority = 9999;
      }
      if (!column.hasOwnProperty('format')) {
        column.format = (val, row) => FMT(val, 2);
      }
      return column;
    });
    return columns.sort((a, b) => {
      return a.priority - b.priority || a.label.localeCompare(b.label)
    });
}


export default defineComponent({
  name: "PageStrategyExplorer",
  components: {
    GameSelect,
    ModelSelect,
    SamplerSettings
  },
  mounted() {
  },
  data() {
    return {
      game: null,
      selected_samples: false,
      loading_samples: false,
      num_samples: 100,
      seed: 1234,
      selector: {},
      pagination: { sortBy: 'num_plays', descending: true, rowsPerPage: 100 },
      history_prefix: [],
    };
  },
  computed: {
    players() {
      return _.range(this.game.num_players)
    },
    readyForSamples() {
      return Object.keys(this.selector).length > 0;
    },
    selected_path() {
      let nodes = [{'pretty_str': '(root)'}];
      for (let i = 0; i < this.history_prefix.length; i++) {
        let action_ids = this.actions[i].map(node => node.action);
        let idx = action_ids.indexOf(this.history_prefix[i]);
        nodes.push(this.actions[i][idx]);
      }
      return nodes;
    },
    previous_nodes() {
      return this.selected_path.map((node, i) => ({
        ...formatNode(node),
      }));
    },
    next_nodes() {
      return this.actions[this.history_prefix.length].map(node => ({
        ...formatNode(node),
      }));
    },
    next_visible_columns() {
      let fields = Object.keys(this.next_nodes[0]);
      return _.filter(fields, c => HIDDEN_COLUMNS.indexOf(c) === -1);
    },
    next_columns() {
      let fields = Object.keys(this.next_nodes[0]);
      return columnifier(fields, true);
    },
    previous_visible_columns() {
      let fields = Object.keys(this.previous_nodes[this.previous_nodes.length - 1]);
      return _.filter(fields, c => HIDDEN_COLUMNS.indexOf(c) === -1);
    },
    previous_columns() {
      let fields = Object.keys(this.previous_nodes[this.previous_nodes.length - 1]);
      return columnifier(fields, false);
    },
    ...mapState({
      actions: (state) => (state.samples_from_state ? JSON.parse(JSON.stringify(state.samples_from_state)) : []),
    }),
  },
  methods: {
    onGameSelected(game) {
      this.game = game.value;
    },
    onClickPreviousActionRow(evt, row, idx) {
      this.history_prefix = this.history_prefix.slice(0, idx);
      this.POP_SAMPLES_FROM_STATE({depth: idx+1});
    },
    onClickNextActionRow(evt, row) {
      if (row.type !== 2) {
        this.history_prefix.push(row.action);
        this.getSamples();
      }
    },
    onSelectionUpdated(evt) {
      let {player, ...data} = evt;
      this.selector[player] = data;
    },
    onSampleParamsChanged(evt) {
      this.seed = evt.seed;
      this.num_samples = evt.num_samples;
    },
    onClickRun() {
      this.history_prefix = [];
      this.POP_SAMPLES_FROM_STATE({depth: 0});
      this.getSamples();
    },
    getSamples() {
      this.selected_samples = true;
      this.loading_samples = true;
      
      let url_params = {num_samples: this.num_samples, seed: this.seed, history_prefix: this.history_prefix}
      for (let player in Object.keys(this.selector)) {
        url_params[`player_${player}_checkpoint_pk`] = this.selector[player].checkpoint;
        if (this.selector[player].response !== null) {
          url_params[`player_${player}_br_pk`] = this.selector[player].response;
        }
      }
      let gamePk = this.game.id;
      this.ADD_SAMPLES_FROM_STATE({gamePk, url_params}).then((data) => {
        this.loading_samples = false;
      });
    },
    ...mapActions(["POP_SAMPLES_FROM_STATE", "ADD_SAMPLES_FROM_STATE"]),
  },
});
</script>
