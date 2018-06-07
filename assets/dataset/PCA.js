import React from 'react';


class PCA extends React.Component {

    constructor(props) {
        super(props);

        var color_by = this.props.plot['color_options'][0];

        var points = this.props.plot['points'].concat(this.props.user_data);
        var point_tags = {};
        for (var i = 0; i < points.length; i++) {
            var point = points[i];
            for (var key in point['tags']) {
                if (!point_tags.hasOwnProperty(key)) point_tags[key] = {};
                var tag = point['tags'][key];
                if (!point_tags[key].hasOwnProperty(tag)) point_tags[key][tag] = {
                    color: point['colors'][key],
                    visibility: true,
                };
            }
        }

        this.state = {
            color_by: color_by,
            point_tags: point_tags,
        };
    }

    plotPCA() {

        var data = [];
        var x = [], y = [], z = [], names = [], colors = [];

        var points = this.props.plot['points'].concat(this.props.user_data);
        for (var i = 0; i < points.length; i++) {

            var point = points[i];

            let visible = false;
            for (var key in point.tags) {
                let tag = point.tags[key];
                if (this.state.point_tags[key][tag].visibility) visible = true;
            }

            if (visible) {
                if (point['colors'].hasOwnProperty(this.state.color_by)) {
                    var color = point['colors'][this.state.color_by];
                } else if (point['colors'].hasOwnProperty('Default')) {
                    var color = point['colors']['Default'];
                } else {
                    var color = '#A9A9A9';
                }

                x.push(point['transformed_values'][0]);
                y.push(point['transformed_values'][1]);
                z.push(point['transformed_values'][2]);
                names.push(point['dataset_name']);
                colors.push(color);
            }
        }

        data.push({
            x: x,
            y: y,
            z: z,
            mode: 'markers',
            text: names,
            point_data: point,
            hoverinfo: 'text',
            marker: {
                size: 12,
                color: colors,
                opacity: 0.8,
            },
            type: 'scatter3d',
        })

        var layout = {
            showlegend: false,
            margin: {
                l: 0,
                r: 0,
                b: 0,
                t: 0,
            },
            scene: {
                xaxis:{
                    title: '',
                    showticklabels: false,
                },
                yaxis:{
                    title: '',
                    showticklabels: false,
                },
                zaxis:{
                    title: '',
                    showticklabels: false,
                },
            },
        };

        var config = {
            displaylogo: false,
            displayModeBar: false,
            modeBarButtonsToRemove: [
                'sendDataToCloud',
                'resetCameraLastSave3d',
                'hoverClosest3d',
            ],
        };

        Plotly.react('plot', data, layout, config);
    }

    drawPlotlyVariance(){

        var data = [{
            y: ['PC 1', 'PC 2', 'PC 3'].reverse(),
            x: this.props.explained_variance.reverse(),
            type: 'bar',
            orientation: 'h',
        }];

        var layout = {
            margin: {
                l: 50,
                r: 50,
                b: 50,
                t: 50,
                pad: 4,
            },
        };

        var options = {
            displayModeBar: false,
        };

        Plotly.newPlot('variance_plot', data, layout, options);
    }

    drawPlotlyComponent(component, div){
        var labels = [],
            values = [];
        for(var i = 0; i < component.length; i++){
            labels.push(component[i][0]);
            values.push(component[i][1]);
        }

        var data = [{
            y: labels.reverse(),
            x: values.reverse(),
            type: 'bar',
            orientation: 'h',
        }];

        var layout = {
            yaxis: {
                tickfont: {
                    size: 10,
                },
            },
            autosize: false,
            width: $('#tabs').width(),
            height: $('#tabs').height(),
            margin: {
                l: 50,
                r: 50,
                b: 50,
                t: 50,
                pad: 4,
            },
        };

        Plotly.newPlot(div, data, layout);
    }

    removePlotly(divElement){
        $(divElement).empty();
    }

    changeColor(event){
        this.setState({color_by: event.target.value});
    }

    invertSelection(event){
        let point_tags = Object.assign({}, this.state.point_tags);
        let group = event.target.getAttribute('data-group');

        for (var tag in point_tags[group]) {
            let visibility = point_tags[group][tag].visibility;
            point_tags[group][tag].visibility = !visibility;
        }

        this.setState({point_tags});
    }

    updatePointVisibility(event){
        let point_tags = Object.assign({}, this.state.point_tags);
        let group = event.target.getAttribute('data-group');

        point_tags[group][event.target.value].visibility = event.target.checked;

        this.setState({point_tags});
    }

    componentDidMount(){

        var $color_select = $(this.refs.color_select);
        for (let i in this.props.plot['color_options']) {
            $color_select.append('<option val="' + i + '">' + this.props.plot['color_options'][i] + '</option>');
        }

        this.plotPCA();
        this.drawPlotlyVariance();
        this.drawPlotlyComponent(this.props.components[0], 'pc_1');
        this.drawPlotlyComponent(this.props.components[1], 'pc_2');
        this.drawPlotlyComponent(this.props.components[2], 'pc_3');
    }

    componentDidUpdate(){
        this.plotPCA();
    }

    cleanDiv(div_id){
        $('#' + div_id).empty();
    }

    componentWillUnmount(){
        this.cleanDiv('plot');
        this.cleanDiv('variance_plot');
        this.cleanDiv('pc_1');
        this.cleanDiv('pc_2');
        this.cleanDiv('pc_3');
    }

    render(){

        var point_dropdowns = Object.keys(this.state.point_tags)
                                    .sort(function(a, b) {
                                        return a.toLowerCase().localeCompare(b.toLowerCase());
                                    }).map((key) => {
            var list = Object.keys(this.state.point_tags[key])
                             .sort(function(a, b) {
                                 return a.toLowerCase().localeCompare(b.toLowerCase());
                             }).map((tag) =>
                <span key={tag}>
                    <input
                        key={tag}
                        type='checkbox'
                        data-group={key}
                        value={tag}
                        checked={this.state.point_tags[key][tag].visibility}
                        onChange={this.updatePointVisibility.bind(this)}>
                    </input>
                    <label
                        style={{display: 'inline'}}>
                        <span
                            className="glyphicon glyphicon-stop"
                            style={{color: this.state.point_tags[key][tag].color}}>
                        </span>
                        {tag}
                    </label>
                    <br></br>
                </span>
            );
            return <div key={key} className="dropdown">
                <button
                    className="btn btn-default btn-block dropdown-toggle"
                    type="button"
                    data-toggle="dropdown">
                        Select points by {key.toLowerCase()}&nbsp;
                        <span className="caret"></span>
                </button>
                <ul className="dropdown-menu pre-scrollable" ref="point_visibility_dropdown">
                    <input
                        type="button"
                        className="btn btn-primary"
                        data-group={key}
                        value="Invert selection"
                        onClick={this.invertSelection.bind(this)}>
                    </input>
                    <br></br>
                    {list}
                </ul>
            </div>
        });

        return <div>
            <div className='pca' style={{paddingTop: '20px'}}>
                <h2>Plot</h2>
                <div className='row'>
                    <div className='col-sm-9' style={{border: '1px solid black'}}>
                        <div ref='plot' id='plot'></div>
                    </div>
                    <div className='col-sm-3'>
                        <div className='form-group'>
                            <label htmlFor='color_select'>Color by:</label>
                            <select className='form-control'
                                id='color_select'
                                ref='color_select'
                                onChange={this.changeColor.bind(this)}
                                value={this.state.color_by}>
                            </select>
                        </div>
                        {point_dropdowns}
                    </div>
                </div>
            </div>
            <div style={{paddingTop: '20px'}}>
                <h2>Components</h2>
                <ul className='nav nav-tabs' style={{paddingTop: '10px'}}>
                    <li className='active'><a data-toggle='tab' href='#variance_plot_tab'>Variance ratios</a></li>
                    <li><a data-toggle='tab' href='#pc_1_tab'>Principle component 1</a></li>
                    <li><a data-toggle='tab' href='#pc_2_tab'>Principle component 2</a></li>
                    <li><a data-toggle='tab' href='#pc_3_tab'>Principle component 3</a></li>
                </ul>
                <div className='tab-content' id='tabs'>
                    <div id='variance_plot_tab' className='tab-pane fade in active'>
                        <div id='variance_plot'></div>
                    </div>
                    <div id='pc_1_tab' className='tab-pane fade'>
                        <div id='pc_1'></div>
                    </div>
                    <div id='pc_2_tab' className='tab-pane fade'>
                        <div id='pc_2'></div>
                    </div>
                    <div id='pc_3_tab' className='tab-pane fade'>
                        <div id='pc_3'></div>
                    </div>
                </div>
            </div>
        </div>;
    }
}

PCA.defaultProps = {
    user_data: [],
};

PCA.propTypes = {
    plot: React.PropTypes.object.isRequired,
    explained_variance: React.PropTypes.array.isRequired,
    components: React.PropTypes.array.isRequired,
    user_data: React.PropTypes.array.isRequired,
};

export default PCA;
