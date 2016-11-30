import React from 'react';
import MetaPlot from './MetaPlot';

import './SmallDataView.css';


class SmallRecommendedDataView extends React.Component {
    componentDidMount(){
        
        let star_glyph = $(this.refs.star_glyph),
            add_favorite_url = this.props.add_favorite_url,
            remove_favorite_url = this.props.remove_favorite_url,
            hide_recommendation_url = this.props.hide_recommendation_url;

		$(this.refs.favorite_button).on('click', function () {

            if (star_glyph.hasClass('glyphicon-star-empty')) {
                star_glyph.removeClass('glyphicon-star-empty')
                star_glyph.addClass('glyphicon-star')

                let favorite_count = parseInt($('#favorite_counts').html());
                favorite_count = favorite_count + 1;
                $('#favorite_counts').html(favorite_count);

                $.ajax({url: add_favorite_url});
            }

            else if (star_glyph.hasClass('glyphicon-star')) {
                star_glyph.removeClass('glyphicon-star')
                star_glyph.addClass('glyphicon-star-empty')

                let favorite_count = parseInt($('#favorite_counts').html());
                favorite_count = favorite_count - 1;
                $('#favorite_counts').html(favorite_count);

                $.ajax({url: remove_favorite_url});
            }
		});

        $(this.refs.close_button).on('click', function () {
            let recommended_count = parseInt($('#recommended_counts').html());
            recommended_count = recommended_count - 1;
            $('#recommended_counts').html(recommended_count);

            $.ajax({url: hide_recommendation_url});
		});
    }

    render(){
        var id_select = 'panel_' + this.props.meta_data['id'];
        var id_css_select = '#' + id_select;

        return <div className="panel panel-default" id={id_select}>
            <div className="panel-heading">
                <div className="panel-title pull-left"><a href={this.props.dataset_url}>{this.props.meta_data['name']}</a></div>
                <div className="panel-title pull-right">
                    <button type="button" ref="favorite_button" className="panel-close-button"><span ref="star_glyph" className="glyphicon glyphicon-star-empty"></span></button>&nbsp;
                    <button type="button" ref="close_button" className="panel-close-button" data-target={id_css_select} data-dismiss="alert"><span className="glyphicon glyphicon-remove"></span></button>
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
                            <li><b>Antibody:</b> {this.props.meta_data['antibody']}</li>
                            <li>{this.props.meta_data['strand']}</li>
                            {this.props.meta_data['description'] &&
                                <li><b>Description:</b> {this.props.meta_data['description']}</li>}
                        </ul>
                    </div>
                    <div style={{height:"150px"}} className="col-sm-3">
                        <h4 style={{textAlign:"center"}}>Promoters</h4>
                        <MetaPlot
                            data={this.props.promoter_data}
                        />
                    </div>
                    <div style={{height:"150px"}} className="col-sm-3">
                        <h4 style={{textAlign:"center"}}>Enhancers</h4>
                        <MetaPlot
                            data={this.props.enhancer_data}
                        />
                    </div>
                </div>
            </div>
            </div>
        </div>;

    }
}

SmallRecommendedDataView.propTypes = {
    meta_data: React.PropTypes.object.isRequired,
    promoter_data: React.PropTypes.object.isRequired,
    enhancer_data: React.PropTypes.object.isRequired,
    dataset_url: React.PropTypes.string.isRequired,
    add_favorite_url: React.PropTypes.string.isRequired,
    remove_favorite_url: React.PropTypes.string.isRequired,
    hide_recommendation_url: React.PropTypes.string.isRequired,
};

export default SmallRecommendedDataView;