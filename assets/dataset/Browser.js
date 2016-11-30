import React from 'react';
import ReactDOM from 'react-dom';


class Browser extends React.Component {

    constructor() {
        super();
        this.state = {
            query: 'chr1:10000000-10001000',
            data: null,
        };
    }

    drawD3(svgElement) {
        var intervals = this.state.data["ambig_intervals"];

        var start = this.state.data["start"],
            end = this.state.data["end"];
        var y_max = Math.max(...intervals.map(function(value) { return value[2]; }));

        var margin = {
            "left": 50,
            "bottom": 50,
            "top": 50,
        };

        var plot_h = this.props.height - margin.bottom - margin.top;
        var plot_w = this.props.width - margin.left;

        var x_scale = d3.scale.linear()
            .domain([start+1, end])
            .range([margin.left, plot_w])
            .clamp(true);

        var y_scale = d3.scale.linear()
            .domain([y_max, 0])
            .range([margin.top, plot_h])
            .clamp(true);

        var context = this.refs.canvas.getContext("2d");

        intervals.forEach(function(d, i) {
            context.beginPath();
            context.rect(x_scale(d[0]), y_scale(d[2]), x_scale(d[1]) - x_scale(d[0]), y_scale(0) - y_scale(d[2]));
            context.fillStyle="black";
            context.fill();
            context.stroke();
            context.closePath();
        });

        var svg = d3.select(svgElement).append("svg")
            .attr("width", this.props.width)
            .attr("height", this.props.height);

        var x_axis = d3.svg.axis().scale(x_scale)
            .orient("bottom")
            .ticks(4);
        var y_axis = d3.svg.axis().scale(y_scale).orient("left")
            .ticks(2);

        svg.append("g").attr("class", "axis").call(x_axis)
            .attr("transform", "translate(0," + plot_h + ")");
        svg.append("g").attr("class", "axis").call(y_axis)
            .attr("transform", "translate(" + margin.left + ",0)");
    }

    removeD3(svgElement){
        $(svgElement).empty();
    }

    componentDidMount() {
        let url = `/network/api/dataset/${this.props.id}/browser_view/?query=${this.state.query}`;
        $.get(url, (d) => {
            this.setState({
                data: d,
            });
            this.drawD3(this.refs.svg);
        });
    }

    componentWillUnmount(){
        this.removeD3(this.refs.svg);
    }

    render(){
        let h = this.props.height,
            w = this.props.width;
        return <div className='browser'>
            <svg ref='svg' style={{height: h, width: w, position:'absolute'}}></svg>
            <canvas ref='canvas' height={h} width={w} style={{position:'absolute'}}></canvas>
        </div>;
    }
}

Browser.propTypes = {
    id: React.PropTypes.number.isRequired,
    height: React.PropTypes.number.isRequired,
    width: React.PropTypes.number.isRequired,
};

export default Browser;
