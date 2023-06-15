<template>
  <q-page class="q-px-md row flex flex-center">
    <div class="col-9">
      <div class="text-h5 q-py-md">Opening Explorer</div>
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
        <q-input
          v-model.number="num_samples"
          type="number"
          label="Number of samples"
          @keydown.enter.prevent="getSamples()"
        />
        <q-input
          v-model.number="seed"
          type="seed"
          label="Random seed"
          @keydown.enter.prevent="getSamples()"
        />
        <br>
        <q-btn
          label="Run"
          icon="model_training"
          color="primary"
          :disabled="!readyForSamples"
          :loading="loading_samples"
          @click="getSamples()"
        />
      </div>

      <br/>
      <div class="q-pa-md shadow-box shadow-5">
        <div class="text-h6">Game Tree</div>
        <template v-if="!selected_samples">
          Select models and press Run to display game tree.
        </template>
        <template v-else>
          <template v-if="loading_samples">
            Sampling game trees...
          </template>
          <template v-else>
            <q-select
              label="Player"
              v-model="selected_player"
              :options="dropdown_players"
              emit-value
            />
            <template v-if="selected_player !== null">
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
import _ from 'lodash' 
import { FMT_STR, FMT } from "../utils.js";

function formatNode(node) {
  // get all properties except children from node
  var { children, ...outcomes } = node;
  
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
  name: "PageOpeningExplorer",
  components: {
    GameSelect,
    ModelSelect,
  },
  mounted() {
  },
  data() {
    return {
      game: null,
      selected_player: null,
      selected_actions: [],
      selected_samples: false,
      loading_samples: false,
      num_samples: 100,
      seed: 1234,
      selector: {},
      pagination: { sortBy: 'num_plays', descending: true, rowsPerPage: 100 },
    };
  },
  computed: {
    players() {
      return _.range(this.game.num_players)
    },
    readyForSamples() {
      return Object.keys(this.selector).length > 0;
    },
    dropdown_players() {
      let players = this.trees.map((tree, i) => ({ label: "Player " + i, value: i }));
      return players;
    },
    selected_path() {
      if (this.selected_player === null) {
        return [];
      } else {
        let node = this.trees[this.selected_player];
        let nodes = [{...node}];
        for (const action_key of this.selected_actions[this.selected_player]) {
          node = node.children[action_key];
          nodes.push({...node});
        }
        return nodes;
      }
    },
    previous_nodes() {
      let prev_node_names = ["(root)"].concat(this.selected_actions[this.selected_player]);
      return this.selected_path.map((node, i) => ({
        action_key: prev_node_names[i],
        ...formatNode(node),
      }));
    },
    next_nodes() {
      let selected_node = this.selected_path[this.selected_path.length - 1];
      if (selected_node.children === null) {
        return [];
      } else {
        let children = selected_node.children;
        let table_nodes = Object.keys(children).map((action_key) => ({
          action_key: action_key,
          ...formatNode(children[action_key]),
        }))
        let conditional_num_plays = _.sum(table_nodes.map(row => row['num_plays']));
        for (const row of table_nodes) {
          row.pct_plays = (row['num_plays'] / conditional_num_plays) * 100;
        }
        return table_nodes;
      }
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
      let fields = Object.keys(this.previous_nodes[0]);
      return _.filter(fields, c => HIDDEN_COLUMNS.indexOf(c) === -1);
    },
    previous_columns() {
      let fields = Object.keys(this.previous_nodes[0]);
      return columnifier(fields, false);
    },
    ...mapState({
      trees: (state) => (state.samples.trees ? JSON.parse(JSON.stringify(state.samples.trees)) : []),
    }),
  },
  methods: {
    onGameSelected(game) {
      this.game = game.value;
    },
    onClickPreviousActionRow(evt, row) {
      let idx = this.selected_actions[this.selected_player].indexOf(row.action_key);
      this.selected_actions[this.selected_player] = this.selected_actions[this.selected_player].slice(0, idx + 1);
    },
    onClickNextActionRow(evt, row) {
      this.selected_actions[this.selected_player].push(row.action_key);
    },
    onSelectionUpdated(evt) {
      let {player, ...data} = evt;
      this.selector[player] = data;
    },
    getSamples() {
      this.selected_samples = true;
      this.loading_samples = true;
      
      let url_params = {num_samples: this.num_samples, seed: this.seed}
      for (let player in Object.keys(this.selector)) {
        url_params[`player_${player}_checkpoint_pk`] = this.selector[player].checkpoint;
        if (this.selector[player].response !== null) {
          url_params[`player_${player}_br_pk`] = this.selector[player].response;
        }
      }
      let gamePk = this.game.id;
      this.GET_SAMPLES({gamePk, url_params}).then((data) => {
        this.loading_samples = false;
        this.selected_actions = this.trees.map((tree) => []);
        this.selected_player = 0;
      });
    },
    ...mapActions(["GET_SAMPLES"]),
  },
});
</script>
