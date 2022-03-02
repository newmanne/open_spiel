<template>
  <apexcharts ref="chart" :height="height" type="heatmap" :options="options" :series="series"></apexcharts>
</template>

<script>
  import { mapState, mapMutations } from 'vuex';
  import VueApexCharts from 'vue-apexcharts';
  import {toPct} from "../../utils";

  export default {
    name: "AllocationsHeatmap",
    components: {
      apexcharts: VueApexCharts,
    },
    props: ['alloc', 'serviceArea', 'showZero'],
    computed: {
      options() {
        const self = this;
        return {
          chart: {
            animations: {
              enabled: false
            },
            events: {
              // click: function(e) {
              dataPointSelection(event, chartContext, config) {
                const index = config.dataPointIndex;
                const series = self.series[config.seriesIndex];
                const bidder = series.name;
                const filter = GroupFilterModal.methods.allocFilter(bidder, self.serviceArea);

                self.ADD_GROUP_FILTERS({
                  key: filter.value,
                  operator: '__exact',
                  value: index
                });
                // console.log(filter);
                self.OPEN_GROUP_FILTERS();
              }
            }
          },
          title: {
            text: this.serviceArea.name,
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
            // dropShadow: {
            //   enabled: true
            // }
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
        // series.reverse();
        // console.log(series);
        // console.log(this.bidders);
        series.sort((x, y) => {
          const bidderX = x.name;
          const bidderY = y.name;
          let indexX = this.bidders.findIndex(x => x.name === bidderX);
          if (indexX === -1) {
            indexX = this.bidders.length;
          }
          let indexY = this.bidders.findIndex(x => x.name === bidderY);
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
        return this.series.map(d => this.bidderColors[d.name])
      },
      ...mapState({
        bidderColors: state => state.example.bidderColors,
        bidders: state => state.example.bidders,
      })
    },
    methods: mapMutations(
      []
    )
  }
</script>

<style scoped>

</style>