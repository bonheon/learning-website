from django.shortcuts import render, get_object_or_404
from .models import Post, Comment
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.views.generic import ListView
from .forms import EmailPostForm, CommentForm, SearchForm
from django.core.mail import send_mail
from django.views.decorators.http import require_POST
from taggit.models import Tag
from django.db.models import Count
from django.contrib.postgres.search import SearchVector, SearchQuery, SearchRank
from django.contrib.postgres.search import TrigramSimilarity

# Create your views here.

class PostListView(ListView) :
    """
    Alternative post list view
    """
    queryset = Post.published.all()
    context_object_name = 'posts'
    paginate_by = 3
    template_name = 'blog/post/list.html'

def post_list(request, tag_slug=None):
    post_list = Post.published.all()
    tag = None
    if tag_slug:
        tag = get_object_or_404(Tag, slug=tag_slug)
        post_list = post_list.filter(tags__in=[tag])
    #페이지당 3개의 게시물로 페이지 매김
    paginator = Paginator(post_list, 3)
    page_number = request.GET.get('page', 1)

    try : 
        posts = paginator.page(page_number)
    except PageNotAnInteger : 
        posts = paginator.page(1)
    except EmptyPage : 
        posts = paginator.page(paginator.num_pages)
    return render(request, 
                  'blog/post/list.html',
                  {'posts': posts,
                   'tag':tag})

def post_detail(request, year, month, day, post):
    post = get_object_or_404(Post,
                             status=Post.Status.PUBLISHED,
                             slug=post,
                             publish__year=year,
                             publish__month=month,
                             publish__day=day)
    # 이 글의 active 댓글 목록
    comments = post.comments.filter(active=True)
    # 사용자가 댓글을 달 수 있는 폼
    form = CommentForm()
    # 유사한 게시물들의 목록
    post_tags_ids = post.tags.values_list('id', flat=True)
    similar_posts = Post.published.filter(tags__in=post_tags_ids).exclude(id=post.id)
    similar_posts = similar_posts.annotate(same_tags=Count('tags')).order_by('-same_tags','-publish')[:4]
    return render(request, 
                  'blog/post/detail.html',
                  {'post': post,
                   'comments': comments,
                   'form': form,
                   'similar_posts': similar_posts})

def post_share(request, post_id) :
    # id로 게시물 조회
    post = get_object_or_404(Post, id=post_id, status=Post.Status.PUBLISHED)
    sent = False
    if request.method == 'POST' : 
        # 폼이 제출되었다면
        form = EmailPostForm(request.POST)
        if form.is_valid() :
            #유효성 검사를 통과한 폼 필드들
            cd = form.cleaned_data 
            post_url = request.build_absolute_uri(post.get_absolute_url())
            subject = f"{cd['name']} recommends you read " \
                      f"{post.title}"
            message = f"Read {post.title} at {post_url}\n\n" \
                      f"{cd['name']}\'s comments: {cd['comments']}"
            send_mail(subject, message, 'bonuni@gmail.com', [cd['to']])
            sent=True
    else :
        form = EmailPostForm()
    return render(request, 'blog/post/share.html', {'post':post, 'form':form, 'sent':sent})

@require_POST
def post_comment(request, post_id):
    post = get_object_or_404(Post, id=post_id, status=Post.Status.PUBLISHED)
    comment = None
    # 댓글이 달림
    form = CommentForm(data=request.POST)
    if form.is_valid():
        # 데이터베이스에 저장하지 않고 Comment 객체 만들기
        comment = form.save(commit=False)
        # 댓글에 게시물 할당하기
        comment.post = post
        # 댓글을 데이터베이스에 저장
        comment.save()
    
    return render(request, 'blog/post/comment.html',
                            {'post':post,
                             'form':form,
                             'comment':comment})
    
def post_search(request):
    form = SearchForm()
    query = None
    results = []
    
    if 'query' in request.GET :
        form = SearchForm(request.GET)
        if form.is_valid():
            query = form.cleaned_data['query']
            search_vector = SearchVector('title', weight='A') + SearchVector('body', weight='B')
            search_query = SearchQuery(query)
            # results = Post.published.annotate(search = search_vector,
            #                                   rank=SearchRank(search_vector, search_query)
            #                                   ).filter(rank__gte=0.3).order_by('-rank')
            results = Post.published.annotate(similarity = TrigramSimilarity('title', query),
                                              ).filter(similarity__gt=0.1).order_by('-similarity')
            
    return render(request,
                  'blog/post/search.html',
                  {'form':form,
                   'query':query,
                   'results':results})
    
    