<template>
  <apexcharts ref="chart" :height="height" type="heatmap" :options="options" :series="series"></apexcharts>
</template>

<script>
  
  import { createNamespacedHelpers } from "vuex";
  const { mapState, mapActions } = createNamespacedHelpers("auctions");
  import VueApexCharts from "vue3-apexcharts";
  import {toPct} from "../../utils";

  export default {
    name: "AllocationsHeatmap",
    components: {
      apexcharts: VueApexCharts,
    },
    props: ['alloc', 'serviceArea', 'showZero', 'bidders'],
    computed: {
      options() {
        const self = this;
        return {
          chart: {
            animations: {
              enabled: false
            },
          },
          title: {
            text: this.serviceArea,
          },
          dataLabels: {
            formatter: v => {
              if (v === 0) {
                return '0%';
              } else if (v < 0.01) {
                return '<1%';
              }
              return toPct(v);
            },
            style: {
              // colors: this.colors // A color per bidder
              colors: Object.keys(this.alloc).map(() => '#333333')
            },
          },
          colors: this.colors,
          xaxis: {
            title: {
            },
            labels: {
            },
            axisTicks: {
              show: false
            },
          },
          tooltip: {
            enabled: false
          }
        }
      },
      height() {
        if (!this.alloc) {
          return 180;
        }
        return Object.keys(this.alloc).length * 60;
      },
      series() {
        if (!this.alloc) {
          return [];
        }
        let series = Object.entries(this.alloc).map(
          ([key, value]) => {
            return {
              name: key,
              data: Object.entries(value).filter(([k, v]) => this.showZero || k > 0).map(([k, v]) => ({x: k, y: v}))
            }
          }
        );
        series.sort((x, y) => {
          const bidderX = x;
          const bidderY = y;
          let indexX = this.bidders.findIndex(x => x === bidderX);
          if (indexX === -1) {
            indexX = this.bidders.length;
          }
          let indexY = this.bidders.findIndex(x => x === bidderY);
          if (indexY === -1) {
            indexY = this.bidders.length;
          }
          return indexY - indexX;
        });
        return series;
      },
      colors() {
        if (!this.series) {
          return [];
        }
        let c = this.series.map(d => this.bidderColors[d.name]);
        return c;
      },
      ...mapState({
        bidderColors: state => state.bidderColors,
      })
    },
  }
</script>

<style scoped>

</style>