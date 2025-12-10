import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/__docusaurus/debug',
    component: ComponentCreator('/__docusaurus/debug', '5ff'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/config',
    component: ComponentCreator('/__docusaurus/debug/config', '5ba'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/content',
    component: ComponentCreator('/__docusaurus/debug/content', 'a2b'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/globalData',
    component: ComponentCreator('/__docusaurus/debug/globalData', 'c3c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/metadata',
    component: ComponentCreator('/__docusaurus/debug/metadata', '156'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/registry',
    component: ComponentCreator('/__docusaurus/debug/registry', '88c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/routes',
    component: ComponentCreator('/__docusaurus/debug/routes', '000'),
    exact: true
  },
  {
    path: '/404',
    component: ComponentCreator('/404', '5c5'),
    exact: true
  },
  {
    path: '/docs',
    component: ComponentCreator('/docs', 'fb4'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', '5ec'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', 'fc8'),
            routes: [
              {
                path: '/docs/assets/diagrams/placeholder',
                component: ComponentCreator('/docs/assets/diagrams/placeholder', '3db'),
                exact: true
              },
              {
                path: '/docs/assets/images/placeholder',
                component: ComponentCreator('/docs/assets/images/placeholder', '475'),
                exact: true
              },
              {
                path: '/docs/case-studies/boston-dynamics-atlas',
                component: ComponentCreator('/docs/case-studies/boston-dynamics-atlas', '65d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/case-studies/figure-ai',
                component: ComponentCreator('/docs/case-studies/figure-ai', 'cfa'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/case-studies/tesla-optimus',
                component: ComponentCreator('/docs/case-studies/tesla-optimus', 'bc2'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/getting-started/installation',
                component: ComponentCreator('/docs/getting-started/installation', '267'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/getting-started/overview',
                component: ComponentCreator('/docs/getting-started/overview', '3b4'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/getting-started/quickstart',
                component: ComponentCreator('/docs/getting-started/quickstart', '1cd'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/intro',
                component: ComponentCreator('/docs/intro', '61d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module-1-fundamentals/chapter-1-introduction',
                component: ComponentCreator('/docs/modules/module-1-fundamentals/chapter-1-introduction', '611'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module-1-fundamentals/chapter-2-history',
                component: ComponentCreator('/docs/modules/module-1-fundamentals/chapter-2-history', '5bf'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module-1-fundamentals/chapter-3-basic-concepts',
                component: ComponentCreator('/docs/modules/module-1-fundamentals/chapter-3-basic-concepts', '6c1'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module-2-hardware/chapter-4-actuators',
                component: ComponentCreator('/docs/modules/module-2-hardware/chapter-4-actuators', '6b4'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module-2-hardware/chapter-5-sensors',
                component: ComponentCreator('/docs/modules/module-2-hardware/chapter-5-sensors', '44d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module-2-hardware/chapter-6-control-systems',
                component: ComponentCreator('/docs/modules/module-2-hardware/chapter-6-control-systems', 'a4d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module-3-control/chapter-7-motion-planning',
                component: ComponentCreator('/docs/modules/module-3-control/chapter-7-motion-planning', 'd11'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module-3-control/chapter-8-locomotion',
                component: ComponentCreator('/docs/modules/module-3-control/chapter-8-locomotion', '8c0'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module-3-control/chapter-9-balance',
                component: ComponentCreator('/docs/modules/module-3-control/chapter-9-balance', '1d3'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module-4-ai/chapter-10-perception',
                component: ComponentCreator('/docs/modules/module-4-ai/chapter-10-perception', 'acd'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module-4-ai/chapter-11-learning',
                component: ComponentCreator('/docs/modules/module-4-ai/chapter-11-learning', 'b9a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module-4-ai/chapter-12-decision-making',
                component: ComponentCreator('/docs/modules/module-4-ai/chapter-12-decision-making', '556'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module-5-applications/chapter-13-simulation',
                component: ComponentCreator('/docs/modules/module-5-applications/chapter-13-simulation', '7e6'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module-5-applications/chapter-14-real-world-deployment',
                component: ComponentCreator('/docs/modules/module-5-applications/chapter-14-real-world-deployment', 'a02'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module-5-applications/chapter-15-future-directions',
                component: ComponentCreator('/docs/modules/module-5-applications/chapter-15-future-directions', 'ccb'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/modules/module-5-applications/chapter-16-case-studies',
                component: ComponentCreator('/docs/modules/module-5-applications/chapter-16-case-studies', '4f5'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/reference/api-reference',
                component: ComponentCreator('/docs/reference/api-reference', '955'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/reference/code-examples',
                component: ComponentCreator('/docs/reference/code-examples', '201'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/reference/hardware-specs',
                component: ComponentCreator('/docs/reference/hardware-specs', '559'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/tutorials/basic-control-examples',
                component: ComponentCreator('/docs/tutorials/basic-control-examples', 'b4a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/tutorials/basic-control-examples/balance-control',
                component: ComponentCreator('/docs/tutorials/basic-control-examples/balance-control', '78f'),
                exact: true
              },
              {
                path: '/docs/tutorials/basic-control-examples/joint-control-basics',
                component: ComponentCreator('/docs/tutorials/basic-control-examples/joint-control-basics', '1ce'),
                exact: true
              },
              {
                path: '/docs/tutorials/basic-control-examples/simple-locomotion',
                component: ComponentCreator('/docs/tutorials/basic-control-examples/simple-locomotion', '412'),
                exact: true
              },
              {
                path: '/docs/tutorials/deployment-guides',
                component: ComponentCreator('/docs/tutorials/deployment-guides', '2ae'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/tutorials/simulation-workflows',
                component: ComponentCreator('/docs/tutorials/simulation-workflows', 'f82'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/',
    component: ComponentCreator('/', '2e1'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
