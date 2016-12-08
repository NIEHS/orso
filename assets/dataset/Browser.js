import React from 'react';
import ReactDOM from 'react-dom';

import './Browser.css';

var MARGINS = {
    "left": 50,
    "bottom": 6,
    "top": 10,
    "right": 0,
};
var HEIGHT = 80;


class Browser extends React.Component {

    constructor(props) {
        super(props);
        this.state = {
            chromosome: 'chr1',
            start: 10000000,
            end: 10001000,
            data: null,
            x_scale: null,
            assembly: null,
            dataset_list: [this.props.id],
        };
    }

    queryToString() {
        return this.state.chromosome + ':' + this.state.start + '-' + this.state.end;
    }

    browserViewUrl(chromosome, start, end, datasets) {
        return `/network/browser/?chr=${chromosome}&start=${start}&end=${end}&datasets=${datasets.join()}`;
    }

    drawPlots() {
        for (let i = 0; i < this.state.data.length; i++) {

            var div = $('<div></div>')
                .css('height', HEIGHT);
            var name = $('<div>' + this.state.data[i]["name"] + '</div>')
                .css('position', 'absolute')
                .css({
                    'padding-left': MARGINS.left + 6,
                    'padding-top': 6,
                });
            var svgElement = $('<svg></svg>')
                .css('position', 'absolute')
                .css({
                    height: HEIGHT,
                    width: $(this.refs.browser_plots).width(),
                });
            var canvas = $('<canvas></canvas>')
                .css('position', 'absolute')
                .prop({
                    height: HEIGHT,
                    width: $(this.refs.browser_plots).width(),
                });

            $(this.refs.browser_plots).append(div);
            div.append(name);
            div.append(svgElement);
            div.append(canvas);

            var intervals = this.state.data[i]["ambig_intervals"];

            var start = this.state.start,
                end = this.state.end;
            var y_max = Math.max(...intervals.map(function(value) { return value[2]; }));

            var plot_h = HEIGHT - MARGINS.bottom;
            var plot_w = this.props.width - MARGINS.left;

            var x_scale = this.state.x_scale;

            var y_scale = d3.scale.linear()
                .domain([y_max, 0])
                .range([MARGINS.top, plot_h])
                .clamp(true);

            var context = canvas[0].getContext("2d");

            intervals.forEach(function(d, i) {
                context.beginPath();
                context.rect(x_scale(d[0]), y_scale(d[2]), x_scale(d[1]) - x_scale(d[0]), y_scale(0) - y_scale(d[2]));
                context.fillStyle="black";
                context.fill();
                context.stroke();
                context.closePath();
            });

            var svg = d3.select(svgElement[0]).append("svg")
                .attr("width", this.props.width)
                .attr("height", HEIGHT);

            var y_axis = d3.svg.axis().scale(y_scale).orient("left")
                .ticks(2);

            svg.append("g").attr("class", "axis").call(y_axis)
                .attr("transform", "translate(" + MARGINS.left + ",0)");
        }
    }

    removeD3(svgElement){
        $(svgElement).empty();
        var context = this.refs.canvas.getContext("2d");
        context.clearRect(0, 0, $(this.refs.canvas).width(), $(this.refs.canvas).height());
    }

    setXScale() {
        var x_scale = d3.scale.linear()
            .domain([this.state.start, this.state.end])
            .range([MARGINS.left, $(this.refs.browser).width() - MARGINS.left - MARGINS.right])
            .clamp(true);

        this.setState({x_scale: x_scale});
    }

    drawXAxis() {
        var svgElement = $('<svg></svg>')
            .prop({
                height: 40,
                width: $(this.refs.browser_plots).width(),
            });
        $(this.refs.x_scale).append(svgElement);

        var svg = d3.select(svgElement[0]).append("svg")
            .attr("width", this.props.width);

        var x_scale = this.state.x_scale;

        var x_axis = d3.svg.axis().scale(x_scale)
            .orient("bottom")
            .ticks(4);

        svg.append("g").attr("class", "axis").call(x_axis)
            .attr("transform", "translate(0," + 0 + ")");

    }

    readAndSaveQueryText(query) {
        if (query) {
            var chromosome = query.split(':')[0],
                start = null,
                end = null;
            if (query.split(':')[1]) {
                start = query.split(':')[1].split('-')[0];
                end = query.split(':')[1].split('-')[1];
                if (start && end) {
                    this.setState({
                        chromosome: chromosome,
                        start: start,
                        end: end,
                    });
                    return true
                }
            }
        } else {
            return true
        }
        return false
    }

    setButtons () {
        let self = this;
        $(this.refs.go_button).on('click', function () {
            let dataset_list = [self.props.id];
            $("input:checkbox:checked").each(function(){
                dataset_list.push(parseInt($(this).val()));
            });
            self.setState({dataset_list: dataset_list});

            if (self.readAndSaveQueryText($(self.refs.query_text).val())) {
                self.draw();
            }
        });
    }

    setDataList() {
        let data_select = $(this.refs.data_select),
            id = this.props.id;
        data_select.append(
            '<li><a href="#">Personal datasets:</a></li>'
        )
        this.props.selectable_datasets['personal'].forEach( function(d) {
            if (id != d.id) {
                data_select.append(
                    '<li><a href="#"><input value=' + d.id + ' type="checkbox"/>&nbsp;&nbsp;' + d.name + '</a></li>'
                );
            }
        });
        data_select.append(
            '<li><a href="#">Favorite datasets:</a></li>'
        )
        this.props.selectable_datasets['favorite'].forEach( function(d) {
            if (id != d.id) {
                data_select.append(
                    '<li><a href="#"><input value=' + d.id + ' type="checkbox"/>&nbsp;&nbsp;' + d.name + '</a></li>'
                );
            }
        });
    }

    drawAssemblyLine() {
        var x_scale = this.state.x_scale;
        var places = (this.state.end - this.state.start).toString().length;
        var unit = 'bases',
            guide_length = Math.pow(10, places - 2);
        if (places > 7) {
            unit = 'Mb';
            guide_length = guide_length / Math.pow(10, 6);
        } else if (places > 4) {
            unit = 'kb';
            guide_length = guide_length / Math.pow(10, 3);
        }
        var x = x_scale(guide_length);
        var w = $(this.refs.browser).width() - MARGINS.left - MARGINS.right;


        $(this.refs.assembly).append(`<i>${this.state.data[0]['assembly']}</i>`);
        var svg = d3.select(this.refs.size_guide).append('svg');

        var lines = [
            {x1: 5, x2: 5, y1: 8, y2: 16},
            {x1: 5, x2: x + 5, y1: 12, y2: 12},
            {x1: x + 5, x2: x + 5, y1: 8, y2: 16},
        ];

        svg.append('g')
            .selectAll('line')
            .data(lines)
            .enter()
            .append('line')
            .attr('x1', (d)=>d.x1)
            .attr('y1', (d)=>d.y1)
            .attr('x2', (d)=>d.x2)
            .attr('y2', (d)=>d.y2)
            .style('stroke', 'black')
            .style('stroke-width', '1');

        svg.append('text')
            .text(guide_length + ' ' + unit)
            .attr('x', 60)
            .attr('y', 16);
    }

    clearDivs() {
        $(this.refs.assembly).empty();
        $(this.refs.size_guide).empty();
        $('#browser_plots').empty();
        $('#x_scale').empty();
    }

    draw() {
        this.clearDivs();
        $.get(this.browserViewUrl(
            this.state.chromosome,
            this.state.start,
            this.state.end,
            this.state.dataset_list
        ), (d) => {
            this.setState({
                data: d,
            });
            this.setXScale();
            this.drawAssemblyLine();
            this.drawPlots();
            this.drawXAxis();
        });
    }

    componentDidMount() {
        this.draw();
        this.setButtons();
        this.setDataList();
    }

    componentWillUnmount(){
        this.removeD3(this.refs.svg);
    }

    render(){
        let h = this.props.height,
            w = this.props.width;
        return <div ref='browser' className='browser'>
            <div className="input-group">
                <div className="input-group-btn">
                    <button type="button" className="btn btn-default dropdown-toggle" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">Select dataset <span className="caret"></span></button>
                    <ul ref='data_select' className="dropdown-menu scrollable-menu">
                    </ul>
                </div>
                <input ref='query_text' type="text" className="form-control" aria-label="..." placeholder={this.queryToString()}></input>
                <span className="input-group-btn">
                    <button ref='go_button' className="btn btn-default" type="button">Go!</button>
                </span>
            </div>
            <div id='assembly_line' className='row' height='40px' style={{'paddingTop':10}}>
                <div ref='assembly' style={{'paddingLeft':MARGINS.left}} className="col-sm-3"></div>
                <div ref='size_guide' style={{'padding':0,'height':20}} className="col-sm-9"></div>
            </div>
            <div id='browser_plots' ref='browser_plots'></div>
            <div id='x_scale' ref='x_scale' style={{'height':40}}></div>
        </div>;
    }
}

Browser.propTypes = {
    id: React.PropTypes.number.isRequired,
    height: React.PropTypes.number.isRequired,
    width: React.PropTypes.number.isRequired,
    selectable_datasets: React.PropTypes.object.isRequired,
    assembly: React.PropTypes.string.isRequired,
};

export default Browser;
