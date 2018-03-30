import React from 'react';
import ReactDOM from 'react-dom';


class DatasetDataView extends React.Component {

    constructor(props) {
        super(props);

        var region_choices = ['--'].concat(Object.keys(this.props.data_lookup));
        this.state = {
            region: '--',
            region_choices: region_choices,
        };
    }

    componentDidMount(){
        // Add genome region options
        for (var i = 0; i < this.state.region_choices.length; i++) {
            $(this.refs.region_select).append(
                '<option val="' + i + '">' + this.state.region_choices[i] + '</option>');
        }
    }

    updateMetaPlot(plot_data){
        var x = [], y = [];

        for (var i = 0; i < plot_data['metaplot']['bin_values'].length; i++) {
            x.push(plot_data['metaplot']['bin_values'][i]);
            y.push(plot_data['metaplot']['metaplot_values'][i]);
        }

        var data = [{
            x: x,
            y: y,
            type: 'scatter',
        }];

        var layout = {
            autosize: true,
            height: $(this.refs.carousel_inner).height(),
            width: $(this.refs.carousel_inner).width(),
            xaxis: {
                tickvals: plot_data['metaplot']['ticks']['tickvals'],
                ticktext: plot_data['metaplot']['ticks']['ticktext'],
            },
            margin: {
                l: 50,
                r: 50,
                b: 80,
                t: 10,
                pad: 4,
            }
        };

        var options = {
        };

        Plotly.react('metaplot', data, layout, options);
    }

    updateFeatureValues(plot_data){
        var x = [], y = [], feature_names = [];

        for (var i = 0; i < plot_data['feature_values']['medians'].length; i++) {
            x.push(plot_data['feature_values']['medians'][i]);
            y.push(plot_data['feature_values']['values'][i]);
            feature_names.push(plot_data['feature_values']['names'][i]);
        }

        var data = [{
            x: x,
            y: y,
            mode: 'markers',
            type: 'scatter',
            text: feature_names,
        }];

        var layout = {
            autosize: true,
            height: $(this.refs.carousel_inner).height(),
            width: $(this.refs.carousel_inner).width(),
            xaxis: {
                type: 'log',
            },
            yaxis: {
                type: 'log',
            },
            margin: {
                l: 50,
                r: 50,
                b: 80,
                t: 10,
                pad: 4,
            },
            hovermode: 'closest',
        };

        Plotly.react('feature_values', data, layout);
    }

    changeRegion(event){
        this.setState({region: event.target.value});
    }

    updateData(event){
        var pks = this.props.data_lookup[this.state.region];

        var metaplot_url = `/network/api/metaplot/${pks['metaplot']}/`;
        var feature_values_url = `/network/api/feature-values/${pks['feature_values']}/`;

        $.get(metaplot_url, function(data) {
            this.updateMetaPlot(data);
        }.bind(this));
        $.get(feature_values_url, function(data) {
            this.updateFeatureValues(data);
        }.bind(this));
    }

    render(){
        return <div>
        <nav className="navbar navbar-default">
            <div className="collapse navbar-collapse" id="bs-example-navbar-collapse-1">
                <form className="navbar-search">
                    <div>
                        <p className="col-xs-2 navbar-text text-right">Genome Region</p>
                        <div className="col-xs-3">
                            <select ref='region_select'
                                className="form-control navbar-btn"
                                onChange={this.changeRegion.bind(this)}
                                value={this.state.region}>
                            </select>
                        </div>
                        <button type="button"
                            className="btn btn-primary navbar-btn col-xs-1 col-xs-offset-4"
                            onClick={this.updateData.bind(this)}>
                        Go
                        </button>
                    </div>
                </form>
            </div>
        </nav>
        <div ref='data_container' className="row">
            <div ref='metaplot_container' className="col-sm-6">
                <div ref='metaplot' id='metaplot'></div>
            </div>
            <div ref='feature_values_container' className="col-sm-6">
                <div ref='feature_values' id='feature_values'></div>
            </div>
        </div>
    </div>
    }
}

DatasetDataView.propTypes = {
    data_lookup: React.PropTypes.object.isRequired,
};

export default DatasetDataView;
