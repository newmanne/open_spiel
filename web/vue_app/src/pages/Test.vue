<template>
  <q-page class="flex flex-center">
    <div class="q-pa-md" style="max-width: 300px">
      <game-select></game-select>
    </div>
  </q-page>
</template>

<script>
import { defineComponent } from "vue";
import { createNamespacedHelpers } from "vuex";
const { mapState, mapActions } = createNamespacedHelpers("auctions");
import GameSelect from "../components/GameSelect.vue";

export default defineComponent({
  name: "PageTest",
  components: {
    GameSelect,
  },
  data() {
    return {};
  },
  mounted() {},
  computed: mapState({}),
  methods: {
    ...mapActions(["GET_EXPERIMENTS", "GET_RUNS", "GET_CHECKPOINTS"]),
  },
});
</script>
