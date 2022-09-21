<template>
  <div>
    <q-input
      v-model.number="num_samples"
      type="number"
      label="Number of samples"
      @update:model-value="getSamples"
    />
    <q-input
      v-model.number="seed"
      type="seed"
      label="Random seed"
      @update:model-value="getSamples"
    />
    <br />
  </div>
</template>

<script>
import { defineComponent } from "vue";
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
    getSamples() {
      sample_params = {
        num_samples: this.num_samples,
        seed: this.seed,
      };
      this.$emit('input', sample_params);
    }
  }
});
</script>

