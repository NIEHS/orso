/*eslint-env node*/

var webpack = require('webpack'),
    WebpackDevServer = require('webpack-dev-server'),
    config = require('./webpack.config.dev.js');

// deploy
new WebpackDevServer(webpack(config), {
    publicPath: config.output.publicPath,
    hot: true,
    quiet: true,
    historyApiFallback: true,
    stats: { colors: true },
}).listen(config.devPort, '0.0.0.0', function (err, result) {
    if (err){
        console.log(err);
    }
});
