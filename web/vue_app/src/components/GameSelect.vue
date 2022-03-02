<template>
  <q-select v-model="game" :options="games" label="Game" v-if="loaded" @update:model-value="gameSelected"/>
</template>

<script>
import { defineComponent, ref } from "vue";
import { createNamespacedHelpers } from "vuex";
const { mapState, mapActions } = createNamespacedHelpers("auctions");

export default defineComponent({
  name: "GameSelect",
  data() {
    return {
      game: null,
      loaded: false,
    }
  },
  mounted() {
    this.GET_GAMES().then(_ => this.loaded = true);
  },
  computed: mapState({
    games: (state) => state.games.map(e => ({ label: e.name, value: e })),
  }),
  methods: {
    gameSelected() {
      this.$emit('input', this.game);
    },
    ...mapActions(["GET_GAMES"]),
  },
});
</script>
