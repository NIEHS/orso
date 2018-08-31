import React from 'react';


class BarChart extends React.Component {

    drawChart(){
        var data = [{
            x: Object.keys(this.props.data),
            y: Object.values(this.props.data),
            type: 'bar',
            hoverinfo: 'none',
        }];

        var layout = {
            title: this.props.layout['title'],

            autosize: false,
            height: this.props.layout['height'],
            width: this.props.layout['width'],
            margin: this.props.layout['margins'],
        };
        if (typeof this.props.layout['xaxis'] !== 'undefined') {
            layout['xaxis'] = this.props.layout['xaxis'];
        }
        if (typeof this.props.layout['yaxis'] !== 'undefined') {
            layout['yaxis'] = this.props.layout['yaxis'];
        }
        if (typeof this.props.layout['titlefont'] !== 'undefined') {
            layout['titlefont'] = this.props.layout['titlefont'];
        }

        var options = {
            displaylogo: false,
            displayModeBar: false,
        };

        Plotly.newPlot(this.chartId(), data, layout, options);
    }

    clearChart(){
        $(this.refs.chart).empty();
    }

    componentDidMount(){
        this.drawChart();
    }

    componentWillUnmount(){
        this.clearChart();
    }

    chartId(){
        return '_pie_' + this.props.id;
    }

    render(){
        let h = this.props.height,
            w = this.props.width;
        return <div ref='chart' id={this.chartId()}></div>;
    }
}

BarChart.defaultProps = {
    layout: {},
};

BarChart.propTypes = {
    data: React.PropTypes.object.isRequired,
    id: React.PropTypes.oneOfType([
        React.PropTypes.string,
        React.PropTypes.number,
    ]).isRequired,

    layout: React.PropTypes.object,
};

export default BarChart;
