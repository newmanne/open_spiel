<template>
  <div>
    <span><b>Models</b></span>
    <div class="q-pb-md">
      <game-select @input="onGameSelected" />
    </div>
    <template v-if="game !== null">
      <div class="q-py-md" v-for="player in players" :key="player">
        <span
          ><b>Player {{ player }} model:</b></span
        >
        <model-select
          :game="game"
          :player="player"
          @updateSelection="onSelectionUpdated"
        />
      </div>
    </template>
  </div>
</template>

<script>
import { defineComponent, ref } from "vue";
import { createNamespacedHelpers } from "vuex";
const { mapState, mapActions } = createNamespacedHelpers("auctions");
import GameSelect from "../components/GameSelect.vue";
import ModelSelect from "../components/ModelSelect.vue";

export default defineComponent({
  name: "MatchupSelect",
  components: {
    GameSelect,
    ModelSelect
  },
  data() {
    return {
        game: null,
    };
  },
  mounted() {},
  computed: {
    players() {
      return _.range(this.game.num_players)
    },
    ...mapState({})
  },
  methods: {
    onGameSelected(game) {
      this.game = game.value;
    },
    ...mapActions(["GET_GAMES"]),
  },
});
</script>



