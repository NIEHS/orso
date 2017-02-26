import React from 'react';
import MetaPlot from './MetaPlot';

import './MetaPlot.css';


class SmallDataView extends React.Component {

    constructor(props) {
        super(props);
        this.state = {is_favorite: (props.meta_data['is_favorite'] === 'true')};
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
    }

    render(){
        var id_select = 'panel_' + this.props.meta_data['id'];
        var id_css_select = '#' + id_select;
        for (var i = 0; i < this.props.plot_data.length; i++) {
            if (this.props.plot_data[i]['regions'] == 'Promoters') {
                var promoter_metaplot = this.props.plot_data[i]['metaplot'];
            } else if (this.props.plot_data[i]['regions'] == 'Enhancers') {
                var enhancer_metaplot = this.props.plot_data[i]['metaplot'];
            }
        }

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
                    <div style={{height:"200px"}} className="col-sm-6">
                        <ul>
                            <li><b>Data type:</b> {this.props.meta_data['data_type']}</li>
                            <li><b>Cell type:</b> {this.props.meta_data['cell_type']}</li>
                            <li><b>Target:</b> {this.props.meta_data['target']}</li>
                            {this.props.meta_data['description'] &&
                                <li><b>Description:</b> {this.props.meta_data['description']}</li>}
                        </ul>
                    </div>
                    <div style={{height:"150px"}} className="col-sm-3">
                        <h4 style={{textAlign:"center"}}>Promoters</h4>
                        {promoter_metaplot  &&
                            <MetaPlot
                                data={promoter_metaplot}
                            />
                        }
                    </div>
                    <div style={{height:"150px"}} className="col-sm-3">
                        <h4 style={{textAlign:"center"}}>Enhancers</h4>
                        {enhancer_metaplot &&
                            <MetaPlot
                                data={enhancer_metaplot}
                            />
                        }
                    </div>
                </div>
            </div>
            </div>
        </div>;

    }
}

SmallDataView.propTypes = {
    meta_data: React.PropTypes.object.isRequired,
    plot_data: React.PropTypes.array.isRequired,
    urls: React.PropTypes.object.isRequired,

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
