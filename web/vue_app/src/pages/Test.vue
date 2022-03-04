<template>
  <q-page class="flex flex-center">
    <iframe :srcdoc="bokeh_js" style="min-width: 1000px; min-height: 600px;"></iframe>
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
    // GameSelect,
  },
  data() {
    return {
      bokeh_js: '',
    };
  },
  computed: {
    // bokeh_js() {
    //   return this.bokeh_js_text;
    // }
  },
  mounted() {
    this.GET_TRAJECTORY_PLOT({runPk: 32}).then(data => {
      this.bokeh_js = data.bokeh_js;
    });
  },
  computed: mapState({}),
  methods: {
    ...mapActions(["GET_TRAJECTORY_PLOT"]),
  },
});
</script>
