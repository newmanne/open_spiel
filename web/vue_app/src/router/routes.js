const routes = [
  {
    path: "/",
    component: () => import("layouts/MainLayout.vue"),
    children: [
      { path: "", component: () => import("pages/Index.vue")},
      { path: "opening_explorer/", name: 'opening-explorer', component: () => import("pages/OpeningExplorer.vue")},
      { path: "cluster_explorer/", name: 'cluster-explorer', component: () => import("pages/ClusterExplorer.vue")},
      { path: "trajectory_plots/", name: 'trajectory-plot', component: () => import("pages/TrajectoryPlots.vue")},
      { path: "allocation_heatmaps/", name: 'allocation-heatmap', component: () => import("pages/AllocationHeatmaps.vue")},
      { path: "test/", name: 'test', component: () => import("pages/Test.vue")},
    ],
  },

  // Always leave this as last one,
  // but you can also remove it
  {
    path: "/:catchAll(.*)*",
    component: () => import("pages/Error404.vue"),
  },
];

export default routes;