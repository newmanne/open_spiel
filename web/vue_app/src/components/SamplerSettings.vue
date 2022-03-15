<template>
  <div>
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
    <br />
    <q-btn
      label="Run"
      icon="model_training"
      color="primary"
      :disabled="!readyForSamples"
      :loading="loading_samples"
      @click="getSamples()"
    />
  </div>
</template>

<script>
import { defineComponent, ref } from "vue";
import { createNamespacedHelpers } from "vuex";
const { mapState, mapActions } = createNamespacedHelpers("auctions");

export default defineComponent({
  name: "SamplerSettings",
  data() {
    return {
      num_samples: 100,
      seed: 1234,
    };
  },
  mounted() {},
  computed: mapState({}),
  methods: {
    gameSelected() {},
    ...mapActions(["GET_GAMES"]),
  },
});
</script>

