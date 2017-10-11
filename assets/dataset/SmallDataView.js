import React from 'react';
import MetaPlot from './MetaPlot';
import DatasetCarousel from './DatasetCarousel';

import './MetaPlot.css';
import './DatasetCarousel.css';


class SmallDataView extends React.Component {

    constructor(props) {
        super(props);
        this.state = {
            is_favorite: (props.meta_data['is_favorite'] === 'true'),
        };
    }

    componentDidMount(){

        let add_favorite_url = this.props.urls['add_favorite'],
            remove_favorite_url = this.props.urls['remove_favorite'],
            hide_recommendation_url = this.props.urls['hide_recommendation'];
        let self = this;

        $(this.refs.favorite_button).on('click', function () {
            if (self.state.is_favorite) {
                self.setState({
                    is_favorite: false
                });

                let favorite_count = parseInt($('#favorite_counts').html());
                favorite_count = favorite_count - 1;
                $('#favorite_counts').html(favorite_count);

                $.ajax({url: remove_favorite_url});

            } else {
                self.setState({
                    is_favorite: true
                });

                let favorite_count = parseInt($('#favorite_counts').html());
                favorite_count = favorite_count + 1;
                $('#favorite_counts').html(favorite_count);

                $.ajax({url: add_favorite_url});
            }
        });

        $(this.refs.remove_recommendation_button).on('click', function () {
            let recommended_count = parseInt($('#recommended_counts').html());
            recommended_count = recommended_count - 1;
            $('#recommended_counts').html(recommended_count);

            $.ajax({url: hide_recommendation_url});
        });

        $(this.refs.remove_favorite_button).on('click', function () {
            let favorite_count = parseInt($('#favorite_counts').html());
            favorite_count = favorite_count - 1;
            $('#favorite_counts').html(favorite_count);

            $.ajax({url: remove_favorite_url});
        });

        if ($.isEmptyObject(this.props.plot_data) == false) {
            for (var i = 0; i < this.props.plot_data.length; i++) {
                this.drawMetaplot(this.props.plot_data[i]);
            }
        }

        if (this.props.score && this.props.score_dist) {
            this.drawScoreHistogram();
        }
    }

    drawMetaplot(plot_data){
        var div_number = $(this.refs.carousel_inner).children('div').length,
            plot_name = plot_data['assembly'] + ': ' + plot_data['regions'];

        let $data_element = (div_number == 0) ?
            $('<div id="' + this.props.exp_id + '.' + div_number + '" class="active item" style="text-align:center">' + plot_name + '</div>') :
            $('<div id="' + this.props.exp_id + '.' + div_number + '" class="item" style="text-align:center">' + plot_name + '</div>');
        $(this.refs.carousel_inner).append($data_element);

        let $indicator_element = (div_number == 0) ?
            $('<li data-target="#carouselExampleIndicators" data-slide-to=' + div_number + ' class="active"></li>') :
            $('<li data-target="#carouselExampleIndicators" data-slide-to=' + div_number + '></li>');
        $(this.refs.carousel_indicators).append($indicator_element);

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
            autosize: false,
            height: $(this.refs.carousel_inner).height(),
            width: $(this.refs.carousel_inner).width(),
            xaxis: {
                tickvals: plot_data['metaplot']['ticks']['tickvals'],
                ticktext: plot_data['metaplot']['ticks']['ticktext'],
            },
            margin : {
                l: 50,
                r: 50,
                b: 60,
                t: 10,
                pad: 4,
            }
        };

        var options = {
            displaylogo: false,
            displayModeBar: false,
        };

        Plotly.newPlot('' + this.props.exp_id + '.' + div_number, data, layout, options);
    }

    drawScoreHistogram() {
        var div_number = $(this.refs.carousel_inner).children('div').length,
            plot_name = 'Recommendation Score';

        let $data_element = (div_number == 0) ?
            $('<div id="' + this.props.exp_id + '.' + div_number + '" class="active item" style="text-align:center">' + plot_name + '</div>') :
            $('<div id="' + this.props.exp_id + '.' + div_number + '" class="item" style="text-align:center">' + plot_name + '</div>');
        $(this.refs.carousel_inner).append($data_element);

        let $indicator_element = (div_number == 0) ?
            $('<li data-target="#carouselExampleIndicators" data-slide-to=' + div_number + ' class="active"></li>') :
            $('<li data-target="#carouselExampleIndicators" data-slide-to=' + div_number + '></li>');
        $(this.refs.carousel_indicators).append($indicator_element);

        var trace = {
            x: this.props.score_dist,
            type: 'histogram',
        };
        var data = [trace];

        var layout = {
            autosize: false,
            height: $(this.refs.carousel_inner).height(),
            width: $(this.refs.carousel_inner).width(),
            shapes: [
                {
                    type: 'line',
                    xref: 'x',
                    yref: 'paper',
                    x0: this.props.score,
                    y0: 0,
                    x1: this.props.score,
                    y1: 1,
                    line: {
                        color: 'rgb(0, 0, 0)',
                        width: 3,
                    },
                },
            ],
            margin : {
                l: 50,
                r: 50,
                b: 60,
                t: 10,
                pad: 4,
            },
        };

        Plotly.newPlot('' + this.props.exp_id + '.' + div_number, data, layout);
    }

    render(){
        var id_select = 'panel_' + this.props.meta_data['id'];
        var id_css_select = '#' + id_select;
        var carousel_id = 'metaplot_carousel_' + this.props.exp_id;

        return <div className="panel panel-default" id={id_select}>
            <div className="panel-heading">
                <div className="panel-title pull-left">
                    <a href={this.props.urls['detail']}>{this.props.meta_data['name']}</a>
                </div>
                <div className="panel-title pull-right">
                    {this.props.display_favorite &&
                        <button type="button" ref="favorite_button" className="panel-close-button">
                            &nbsp;
                            {this.state.is_favorite ? (
                                <span className="glyphicon glyphicon-star"></span>
                            ) : (
                                <span className="glyphicon glyphicon-star-empty"></span>
                            )}
                        </button>
                    }
                    {this.props.display_edit &&
                        <a href={this.props.urls['edit']} className="btn panel-close-button">
                            &nbsp;<span className="glyphicon glyphicon-pencil"></span>
                        </a>
                    }
                    {this.props.display_remove_recommendation &&
                        <button type="button" ref="remove_recommendation_button" className="panel-close-button"
                                data-target={id_css_select} data-dismiss="alert">
                            &nbsp;<span className="glyphicon glyphicon-remove"></span>
                        </button>
                    }
                    {this.props.display_remove_favorite &&
                        <button type="button" ref="remove_favorite_button" className="panel-close-button"
                                data-target={id_css_select} data-dismiss="alert">
                            &nbsp;<span className="glyphicon glyphicon-remove"></span>
                        </button>
                    }
                    {this.props.display_delete &&
                        <a href={this.props.urls['delete']} className="btn panel-close-button">
                            &nbsp;<span className="glyphicon glyphicon-trash"></span>
                        </a>
                    }
                </div>
                <div className="clearfix"></div>

            </div>
            <div className="panel-body">
                <div className='small_data_view'>
                    <div className="row">
                        <div style={{height:"200px"}} className="col-sm-4">
                            <ul>
                                <li><b>Assembly:</b> {this.props.meta_data['assemblies'].join(', ')}</li>
                                <li><b>Data type:</b> {this.props.meta_data['data_type']}</li>
                                <li><b>Cell type:</b> {this.props.meta_data['cell_type']}</li>
                                <li><b>Target:</b> {this.props.meta_data['target']}</li>
                                {this.props.meta_data['description'] &&
                                    <li><b>Description:</b> {this.props.meta_data['description']}</li>}
                            </ul>
                        </div>
                        <div style={{height:"200px"}} className="col-sm-8">
                            <div id={carousel_id} className="carousel slide" data-ride="carousel" data-interval="false" style={{height: '100%', width: '100%'}}>
                                <ol ref='carousel_indicators' className="carousel-indicators"></ol>
                                <div ref='carousel_inner' className="carousel-inner" role="listbox" style={{height: '100%', width: '80%', left: '10%'}}></div>
                                <a className="left carousel-control" href={'#' + carousel_id} role="button" data-slide="prev">
                                    <span className="glyphicon glyphicon-chevron-left" aria-hidden="true"></span>
                                    <span className="sr-only">Previous</span>
                                </a>
                                <a className="right carousel-control" href={'#' + carousel_id} role="button" data-slide="next">
                                    <span className="glyphicon glyphicon-chevron-right" aria-hidden="true"></span>
                                    <span className="sr-only">Next</span>
                                </a>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>;
    }
}

SmallDataView.propTypes = {
    exp_id: React.PropTypes.number.isRequired,
    meta_data: React.PropTypes.object.isRequired,
    plot_data: React.PropTypes.array.isRequired,
    urls: React.PropTypes.object.isRequired,

    score_dist: React.PropTypes.array,
    score: React.PropTypes.number,

    display_favorite: React.PropTypes.bool,
    display_edit: React.PropTypes.bool,
    display_delete: React.PropTypes.bool,
    display_remove_recommendation: React.PropTypes.bool,
    display_remove_favorite: React.PropTypes.bool,
};

SmallDataView.defaultProps = {
    display_favorite: false,
    display_edit: false,
    display_delete: false,
    display_remove_recommendation: false,
    display_remove_favorite: false,
};

export default SmallDataView;
