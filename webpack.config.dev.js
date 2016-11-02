/*eslint-env node*/

var config = require('./webpack.base.js'),
    webpack = require('webpack'),
    devPort = 5556;

config.devPort = devPort;

// setup hot-reloading
config.module.loaders[0].loaders.unshift('react-hot');
config.output.publicPath = 'http://localhost:5556/static/bundles/';
config.entry.unshift(
    'webpack-dev-server/client?http://localhost:' + devPort,
    'webpack/hot/only-dev-server'
);
config.plugins.unshift(
    new webpack.HotModuleReplacementPlugin()
);

// setup happypack
// --> disabled for now, malforms error messages in webpack-dashboard
// var HappyPack = require('happypack');
// config.module.loaders[0].happy = { id: 'js' };
// config.plugins.push(new HappyPack({ id: 'js', verbose: false, threads: 4 }));

// setup webpack-dashboard
var Dashboard = require('webpack-dashboard'),
    DashboardPlugin = require('webpack-dashboard/plugin'),
    dashboard = new Dashboard();

config.plugins.push(
    new DashboardPlugin(dashboard.setData)
);

module.exports = config;
